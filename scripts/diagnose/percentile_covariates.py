"""Correlate the get_cost() action-sensitivity percentile against per-episode
covariates, to find out *what* separates reliable planning states from
unreliable ones (see CJEPA_PROJECT.md's "Next investigation plan").

action_sensitivity_check.py established that get_cost() carries a real but
highly state-dependent action-sensitivity signal: mean percentile 43.7 (below
the 50 random-chance mark) but std 30.2 -- 17.2% of episodes are excellent,
10.9% are actively inverted. CEM plans one state at a time and can't detect in
advance which regime a given call falls into. This script extends the same
get_cost()-bypasses-CEM construction (same checkpoint, same real-vs-random
percentile) across more held-out episodes, and additionally records five
per-episode covariates for each window:

  - real_action_magnitude   -- ||real future action|| (z-scored units)
  - block_motion            -- ||block_pose[T_h] - block_pose[0]|| (raw units,
                                full [x, y, angle] pose norm, mixed units by
                                design -- matches the originally queued spec)
  - agent_motion             -- ||pos_agent[T_h] - pos_agent[0]|| (raw units)
  - contact_change           -- |n_contacts[T_h] - n_contacts[0]|
  - history_slot_stability  -- mean Hungarian-matched cosine similarity across
                                this episode's own history-frame pairs

then reports Spearman correlation of each covariate against the percentile
score, plus a reliable-quartile (lowest percentile) vs. unreliable-quartile
(highest percentile) covariate comparison.

`block_pose`/`pos_agent`/`n_contacts` are not in cjepa.yaml's training
`keys_to_load` ([pixels, action, proprio, state]), so this script builds its
own dataset instance with those three columns added -- without z-score
normalizing them, since block_motion/agent_motion are specified in raw units.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.cjepa.cjepa import _hungarian_match_slots

CHECKPOINT = 'cjepa_run2/weights_epoch_5.pt'
N_EPISODES = 200
N_RANDOM_CANDIDATES = 32
SEED = 1234
RAW_COVARIATE_KEYS = ('block_pose', 'pos_agent', 'n_contacts')
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'percentile_covariates_results.csv')


def get_img_preprocessor(source, target, img_size):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def build_dataset_and_splits(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')

    # Add the raw covariate columns on top of training's [pixels, action,
    # proprio, state] -- they ride the same frameskip window as pixels, so
    # each sample's block_pose/pos_agent/n_contacts land on the same 4
    # frame-times as pixels/action.
    original_keys = list(dataset_cfg['keys_to_load'])
    dataset_cfg['keys_to_load'] = original_keys + list(RAW_COVARIATE_KEYS)

    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)

    transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
    ]
    for col in original_keys:
        if col.startswith('pixels'):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    # Deliberately no normalizer for RAW_COVARIATE_KEYS -- block_motion /
    # agent_motion are specified in raw units, not z-scored.
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    return val_set


def sanity_check_n_contacts(val_set, rng, n_preview=5):
    print('\n=== n_contacts raw-value sanity check (before trusting contact_change) ===')
    preview_idxs = rng.choice(len(val_set), size=n_preview, replace=False)
    for i in preview_idxs:
        sample = val_set[int(i)]
        nc = sample['n_contacts'].reshape(-1).numpy()
        print(f'  episode window {int(i)}: n_contacts over window = {nc.tolist()}')


@torch.no_grad()
def history_slot_stability(model, hist_pixels, encoder_dtype):
    """Mean Hungarian-matched cosine similarity across this episode's own
    history-frame pairs.

    Args:
        hist_pixels: (1, T_h, C, H, W)

    Returns:
        float, mean cosine similarity across the T_h-1 consecutive pairs.
    """
    slots = model._extract_slots(hist_pixels.to(encoder_dtype))[0].float()  # (T_h, N, D)
    sims = []
    for t in range(slots.shape[0] - 1):
        a, b = slots[t : t + 1], slots[t + 1 : t + 2]  # (1, N, D)
        b_matched = _hungarian_match_slots(b, a)[0]  # (N, D)
        cos = torch.nn.functional.cosine_similarity(a[0], b_matched, dim=-1)  # (N,)
        sims.append(cos.mean().item())
    return float(np.mean(sims))


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
    sanity_check_n_contacts(val_set, rng)

    idxs = rng.choice(len(val_set), size=N_EPISODES, replace=False)

    rows = []  # each: dict of episode_idx, percentile, and 5 covariates

    for i in idxs:
        sample = val_set[int(i)]
        pixels = sample['pixels'].to('cuda')  # (4, C, H, W)
        action = sample['action'].to('cuda')  # (4, act_dim) -- z-scored blocks

        hist_pixels = pixels[:T_h].unsqueeze(0)  # (1, T_h, C, H, W)
        goal_pixels = pixels[T_h]  # (C, H, W) -- real next-step frame
        hist_action_real = action[: T_h - 1]  # (T_h-1, act_dim)
        real_future_action = action[T_h - 1]  # (act_dim,)

        S = 1 + N_RANDOM_CANDIDATES
        d = real_future_action.shape[-1]

        pixels_batch = hist_pixels.unsqueeze(1).expand(1, S, *hist_pixels.shape[1:]).clone()
        goal_batch = goal_pixels.unsqueeze(0)

        hist_action_batch = (
            hist_action_real.unsqueeze(0).unsqueeze(0).expand(1, S, *hist_action_real.shape)
        ).clone()

        action_candidates = torch.randn(1, S, 1, d, device='cuda', dtype=action.dtype)
        action_candidates[0, 0, 0] = real_future_action

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

        # --- covariates ---
        block_pose = sample['block_pose'].numpy()  # (4, 3) -- [x, y, angle]
        pos_agent = sample['pos_agent'].numpy()  # (4, 2)
        n_contacts = sample['n_contacts'].reshape(-1).numpy()  # (4,)

        real_action_magnitude = float(torch.linalg.norm(real_future_action).item())
        block_motion = float(np.linalg.norm(block_pose[T_h] - block_pose[0]))
        agent_motion = float(np.linalg.norm(pos_agent[T_h] - pos_agent[0]))
        contact_change = float(abs(n_contacts[T_h] - n_contacts[0]))
        stability = history_slot_stability(model, hist_pixels, encoder_dtype)

        rows.append(
            {
                'episode_window_idx': int(i),
                'percentile': percentile,
                'real_action_magnitude': real_action_magnitude,
                'block_motion': block_motion,
                'agent_motion': agent_motion,
                'contact_change': contact_change,
                'history_slot_stability': stability,
            }
        )

    percentiles = np.array([r['percentile'] for r in rows])
    print('\n=== Action-sensitivity diagnostic (get_cost, real vs random) ===')
    print(f'N episodes: {len(percentiles)}, K random candidates per episode: {N_RANDOM_CANDIDATES}')
    print(f'Mean percentile:   {percentiles.mean():.1f}  (0=perfect, 50=random-chance, 100=worst)')
    print(f'Median percentile: {np.median(percentiles):.1f}')
    print(f'Std percentile:    {percentiles.std():.1f}')
    print(f'% episodes real beats median random:  {100 * (percentiles < 50).mean():.1f}%')
    print(f'% episodes real in top decile (<10):  {100 * (percentiles < 10).mean():.1f}%')
    print(f'% episodes real in bottom decile (>90): {100 * (percentiles > 90).mean():.1f}%')

    covariate_names = [
        'real_action_magnitude',
        'block_motion',
        'agent_motion',
        'contact_change',
        'history_slot_stability',
    ]

    # reliable = bottom 25% by percentile (real action beats the most randoms)
    # unreliable = top 25% by percentile (real action beats the fewest randoms)
    order = np.argsort(percentiles)
    q = max(1, len(order) // 4)
    reliable_idx = order[:q]
    unreliable_idx = order[-q:]

    print('\n=== Spearman correlation: covariate vs. percentile ===')
    print(f'{"covariate":<24}{"r":>8}{"p":>10}{"reliable-Q mean":>18}{"unreliable-Q mean":>20}')
    for name in covariate_names:
        values = np.array([r[name] for r in rows])
        r, p = spearmanr(values, percentiles)
        reliable_mean = values[reliable_idx].mean()
        unreliable_mean = values[unreliable_idx].mean()
        print(
            f'{name:<24}{r:>8.3f}{p:>10.4f}{reliable_mean:>18.4f}{unreliable_mean:>20.4f}'
        )
    print(f'(reliable/unreliable quartiles: n={q} episodes each)')

    # persist raw per-episode results for later plotting without a GPU rerun
    header = ['episode_window_idx', 'percentile'] + covariate_names
    with open(RESULTS_CSV, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in header) + '\n')
    print(f'\nRaw per-episode results saved to {RESULTS_CSV}')


if __name__ == '__main__':
    main()
