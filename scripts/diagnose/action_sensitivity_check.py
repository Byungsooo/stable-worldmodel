"""Scaled-up version of the ad hoc get_cost() real-vs-random-action check.

Last session (see TODO_UPSTREAM_FIXES.md items 0c/0d) found that, after fixing
the action/timestep-alignment and history-action bugs, get_cost() started
showing a real but weak action-sensitivity signal -- but that was only ever
checked on 5 held-out episodes ("11th-88th percentile spread"). This script
repeats the same check (bypassing CEM and the env entirely -- pure forward
passes through get_cost()) across many more held-out episodes, so the read is
a real statistic instead of an anecdotal spread.

For each held-out validation window, it builds a single get_cost() batch
containing the real expert action alongside K random Gaussian actions (the
data pipeline z-scores actions, so N(0, 1) candidates are on-distribution),
and records what percentile the real action's cost lands at relative to the
random candidates (0 = beats every random candidate, 50 = indistinguishable
from random, 100 = worse than every random candidate).
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

CHECKPOINT = 'cjepa_run2/weights_epoch_5.pt'
N_EPISODES = 64
N_RANDOM_CANDIDATES = 32
SEED = 1234


def get_img_preprocessor(source, target, img_size):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def build_dataset_and_splits(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)

    transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
    ]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels'):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    return val_set


@torch.no_grad()
def main():
    with hydra.initialize(version_base=None, config_path='../train/config'):
        cfg = hydra.compose(config_name='cjepa')

    val_set = build_dataset_and_splits(cfg)
    print(f'Held-out validation windows available: {len(val_set)}')

    model = swm.wm.utils.load_pretrained(CHECKPOINT)
    model = model.to('cuda').eval()
    model.requires_grad_(False)

    T_h = model.history_len  # 3
    encoder_dtype = next(model.slot_encoder.parameters()).dtype

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(val_set), size=N_EPISODES, replace=False)

    percentiles = []
    real_costs = []
    random_cost_means = []

    for i in idxs:
        sample = val_set[int(i)]
        pixels = sample['pixels'].to('cuda')  # (4, C, H, W)
        action = sample['action'].to('cuda')  # (4, act_dim) -- z-scored blocks

        hist_pixels = pixels[:T_h].unsqueeze(0)  # (1, T_h, C, H, W)
        goal_pixels = pixels[T_h]  # (C, H, W) -- real next-step frame
        hist_action_real = action[: T_h - 1]  # (T_h-1, act_dim) -- blocks 0..T_h-2
        real_future_action = action[T_h - 1]  # (act_dim,) -- block T_h-1

        S = 1 + N_RANDOM_CANDIDATES
        d = real_future_action.shape[-1]

        # (B=1, S, T_h, C, H, W) -- identical history repeated across candidates
        pixels_batch = hist_pixels.unsqueeze(1).expand(1, S, *hist_pixels.shape[1:]).clone()
        goal_batch = goal_pixels.unsqueeze(0)  # (B=1, C, H, W)

        hist_action_batch = (
            hist_action_real.unsqueeze(0).unsqueeze(0).expand(1, S, *hist_action_real.shape)
        ).clone()

        action_candidates = torch.randn(1, S, 1, d, device='cuda', dtype=action.dtype)
        action_candidates[0, 0, 0] = real_future_action  # slot 0 = real action

        info = {
            'pixels': pixels_batch.to(encoder_dtype),
            'goal': goal_batch.to(encoder_dtype),
            'hist_action': hist_action_batch,
        }

        cost = model.get_cost(info, action_candidates)  # (1, S)
        cost = cost[0].float().cpu().numpy()

        cost_real = cost[0]
        cost_random = cost[1:]
        percentile = 100.0 * (cost_random < cost_real).sum() / len(cost_random)

        percentiles.append(percentile)
        real_costs.append(cost_real)
        random_cost_means.append(cost_random.mean())

    percentiles = np.array(percentiles)
    print('\n=== Action-sensitivity diagnostic (get_cost, real vs random) ===')
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
