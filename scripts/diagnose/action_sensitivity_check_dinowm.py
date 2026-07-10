"""DINO-WM (PreJEPA) counterpart to action_sensitivity_check.py.

Same statistic, same methodology (bypass CEM and the env entirely -- pure
forward passes through get_cost(), real action vs K random Gaussian
candidates, record the percentile the real action's cost lands at), run
against the trained PreJEPA checkpoint instead of CJEPA. Intended as a direct
comparison point: see CJEPA_PROJECT.md's 2026-07-06 n=64 numbers
(mean 43.7, median 39.1, std 30.2) for CJEPA on the same statistic.

PreJEPA's get_cost()/rollout() has a different interface from CJEPA's, so
this is not a copy-paste of the original script:

1. Action indexing is frame-paired, not transition-indexed. CJEPA's script
   used `action[T_h - 1]` as the evaluated future action because CJEPA
   indexes actions as *transitions between* the T_h context frames (T_h
   frames need only T_h - 1 "already happened" actions, and the action that
   would transition from the last context frame to the goal is the (T_h)th
   overall, i.e. index T_h - 1). PreJEPA's training (`prejepa.py`'s
   `dinowm_forward`) instead pairs action[i] directly with frame[i] as a
   same-index per-frame input feature to the encoder -- so here the action
   evaluated is `action[T_h]`, the one paired with the goal frame itself,
   and the context actions are `action[:T_h]` (paired 1:1 with the T_h
   context frames, not T_h - 1 of them).

2. get_cost() requires a 'goal' image with an explicit unit time dimension
   (B, T=1, C, H, W), unlike CJEPA's (B, C, H, W) -- PreJEPA's _encode_image
   unconditionally does `rearrange(pixels, 'b t ... -> (b t) ...')`.

3. get_cost()/rollout() cache the goal/init embedding across calls keyed on
   info_dict['id'] / info_dict['step_idx'] (set by the env wrapper during
   real MPC rollouts, where the same env instance calls get_cost repeatedly
   across CEM iterations). This diagnostic calls get_cost() once per
   independent held-out window with no env involved, so 'id'/'step_idx'
   are never supplied -- and the cache attributes are explicitly cleared
   before every call so the (hasattr-gated) cache-hit branch, which would
   otherwise KeyError on the missing keys from the second call onward, is
   never taken.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer

CHECKPOINT = 'pusht_dinov2_small_psmall/weights_epoch_10.pt'
N_EPISODES = 64
N_RANDOM_CANDIDATES = 32
SEED = 1234


def get_img_preprocessor(source, target, img_size):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def build_val_set(cfg):
    """Mirrors prejepa.py's dataset construction exactly, so this is the
    same held-out split (same seed, same train_split) the checkpoint never
    trained on."""
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())  # ['proprio', 'action']
    keys_to_load = ['pixels'] + encoding_keys

    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col) for col in encoding_keys
    ]
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', cfg.image_size),
        *normalizers,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    return val_set


def clear_planning_cache(model):
    """PreJEPA caches goal/init embeddings on the model instance itself,
    keyed on env id/step_idx that this offline diagnostic doesn't have.
    Must be cleared before every independent get_cost() call."""
    for attr in ('_goal_cached_info', '_init_cached_info'):
        if hasattr(model, attr):
            delattr(model, attr)


@torch.no_grad()
def main():
    with hydra.initialize(version_base=None, config_path='../train/config'):
        cfg = hydra.compose(config_name='prejepa')

    val_set = build_val_set(cfg)
    print(f'Held-out validation windows available: {len(val_set)}')

    model = swm.wm.utils.load_pretrained(CHECKPOINT)
    model = model.to('cuda').eval()
    model.requires_grad_(False)

    T_h = model.history_size  # 3
    encoder_dtype = next(model.backbone.parameters()).dtype

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(val_set), size=N_EPISODES, replace=False)

    percentiles = []
    real_costs = []
    random_cost_means = []

    for i in idxs:
        sample = val_set[int(i)]
        pixels = sample['pixels'].to('cuda')  # (T_h+1, C, H, W)
        proprio = sample['proprio'].to('cuda').float()  # (T_h+1, proprio_dim)
        action = sample['action'].to('cuda').float()  # (T_h+1, action_dim) -- z-scored blocks, frame-paired

        hist_pixels = pixels[:T_h]  # (T_h, C, H, W)
        goal_pixels = pixels[T_h]  # (C, H, W) -- real next-step frame
        hist_proprio = proprio[:T_h]  # (T_h, proprio_dim)
        goal_proprio = proprio[T_h]  # (proprio_dim,)
        hist_action = action[:T_h]  # (T_h, action_dim) -- frame-paired context actions
        real_future_action = action[T_h]  # (action_dim,) -- action paired with the goal frame

        S = 1 + N_RANDOM_CANDIDATES
        C, H, W = hist_pixels.shape[1:]
        proprio_dim = hist_proprio.shape[-1]
        d = real_future_action.shape[-1]

        # (B=1, S, T_h, ...) -- identical context repeated across candidates
        pixels_batch = hist_pixels.unsqueeze(0).unsqueeze(0).expand(1, S, T_h, C, H, W).clone()
        proprio_batch = hist_proprio.unsqueeze(0).unsqueeze(0).expand(1, S, T_h, proprio_dim).clone()
        action_batch = hist_action.unsqueeze(0).unsqueeze(0).expand(1, S, T_h, d).clone()

        # (B=1, S, T=1, ...) -- goal needs an explicit unit time dim (see module docstring)
        goal_batch = goal_pixels.view(1, 1, 1, C, H, W).expand(1, S, 1, C, H, W).clone()
        goal_proprio_batch = goal_proprio.view(1, 1, 1, proprio_dim).expand(1, S, 1, proprio_dim).clone()

        # action_sequence passed to get_cost: (B=1, S, T_h + num_pred, d)
        # slots [:T_h] = shared context actions, slot [T_h] = the evaluated candidate
        action_seq = torch.empty(1, S, T_h + model.num_pred, d, device='cuda', dtype=action.dtype)
        action_seq[0, :, :T_h] = hist_action
        action_seq[0, :, T_h:] = torch.randn(S, model.num_pred, d, device='cuda', dtype=action.dtype)
        action_seq[0, 0, T_h] = real_future_action  # slot 0 = real action

        info = {
            'pixels': pixels_batch.to(encoder_dtype),
            'proprio': proprio_batch,
            'action': action_batch,  # overwritten inside rollout(); kept consistent for clarity
            'goal': goal_batch.to(encoder_dtype),
            'goal_proprio': goal_proprio_batch,
        }

        clear_planning_cache(model)
        cost = model.get_cost(info, action_seq)  # (1, S)
        cost = cost[0].float().cpu().numpy()

        cost_real = cost[0]
        cost_random = cost[1:]
        percentile = 100.0 * (cost_random < cost_real).sum() / len(cost_random)

        percentiles.append(percentile)
        real_costs.append(cost_real)
        random_cost_means.append(cost_random.mean())

    percentiles = np.array(percentiles)
    print('\n=== Action-sensitivity diagnostic (get_cost, real vs random) -- DINO-WM ===')
    print(f'N episodes: {len(percentiles)}, K random candidates per episode: {N_RANDOM_CANDIDATES}')
    print(f'Mean percentile:   {percentiles.mean():.1f}  (0=perfect, 50=random-chance, 100=worst)')
    print(f'Median percentile: {np.median(percentiles):.1f}')
    print(f'Std percentile:    {percentiles.std():.1f}')
    print(f'% episodes real beats median random:  {100 * (percentiles < 50).mean():.1f}%')
    print(f'% episodes real in top decile (<10):  {100 * (percentiles < 10).mean():.1f}%')
    print(f'% episodes real in bottom decile (>90): {100 * (percentiles > 90).mean():.1f}%')
    print(f'Raw percentiles: {np.round(percentiles, 1).tolist()}')


if __name__ == '__main__':
    main()
