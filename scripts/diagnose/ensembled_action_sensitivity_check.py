"""Tests the "single-forward-pass estimator variance" branch of
percentile_covariates.py's interpretation guide: does averaging get_cost()
over an ensemble of passes produce a cleaner action-sensitivity signal than
a single pass?

get_cost() is otherwise fully deterministic (confirmed empirically: two
_extract_slots() calls on an identical input return bit-identical tensors)
-- the trained predictor has dropout=0.0 throughout, and the VideoSAUR slot
encoder's RandomInit was already fixed (2026-07-04 session) to use a
hardcoded seed=0 every call specifically to stop unseeded noise from
swamping the cost signal. So literally repeating get_cost() on the same
input and averaging has zero effect; there is no per-call noise left.

The one remaining, architecturally real source of designed-but-currently-
pinned stochasticity is RandomInit's slot-space anchor: Slot Attention is
normally randomly initialized by design, and this project pinned it to a
single fixed draw (seed=0) purely to keep history/goal encodings within one
get_cost() call self-consistent, not because the initialization is
meant to be constant across unrelated calls. This script varies that seed
across an ensemble of M independent passes (same seed used consistently
*within* each pass's history+goal encoding, varied *across* passes),
averages the resulting cost tensor, and recomputes the percentile -- then
compares single-pass (seed=0 only, matching the existing baseline) vs.
ensembled percentile statistics on the *same* held-out episodes used by
action_sensitivity_check.py / percentile_covariates.py (same SEED, same
N_EPISODES, same val_set construction -> identical `idxs` sample).
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
N_EPISODES = 200
N_RANDOM_CANDIDATES = 32
SEED = 1234
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]  # seed=0 alone reproduces the existing single-pass baseline


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
def get_cost_with_seed(model, info, action_candidates, seed):
    # rollout() mutates its info dict in place (sets info['slots'] /
    # info['goal_slots']) and returns the same object -- reusing one dict
    # object across ensemble passes would cache the first pass's encoding
    # and silently skip re-encoding on every subsequent seed. A fresh
    # shallow copy forces re-encoding from 'pixels'/'goal' every pass.
    model.slot_encoder.initializer.seed = seed
    return model.get_cost(dict(info), action_candidates)


@torch.no_grad()
def main():
    with hydra.initialize(version_base=None, config_path='../train/config'):
        cfg = hydra.compose(config_name='cjepa')

    val_set = build_dataset_and_splits(cfg)
    print(f'Held-out validation windows available: {len(val_set)}')

    model = swm.wm.utils.load_pretrained(CHECKPOINT)
    model = model.to('cuda').eval()
    model.requires_grad_(False)

    T_h = model.history_len
    encoder_dtype = next(model.slot_encoder.parameters()).dtype

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(val_set), size=N_EPISODES, replace=False)

    single_pass_percentiles = []
    ensembled_percentiles = []

    for i in idxs:
        sample = val_set[int(i)]
        pixels = sample['pixels'].to('cuda')
        action = sample['action'].to('cuda')

        hist_pixels = pixels[:T_h].unsqueeze(0)
        goal_pixels = pixels[T_h]
        hist_action_real = action[: T_h - 1]
        real_future_action = action[T_h - 1]

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

        costs_per_seed = []
        for seed in ENSEMBLE_SEEDS:
            cost = get_cost_with_seed(model, info, action_candidates, seed)  # (1, S)
            costs_per_seed.append(cost[0].float().cpu().numpy())

        # single-pass baseline = seed 0 alone (matches action_sensitivity_check.py exactly)
        cost_single = costs_per_seed[0]
        cost_real_single = cost_single[0]
        cost_random_single = cost_single[1:]
        single_pass_percentiles.append(
            100.0 * (cost_random_single < cost_real_single).sum() / len(cost_random_single)
        )

        # ensembled = mean cost across all ENSEMBLE_SEEDS passes, then percentile
        cost_ensembled = np.mean(costs_per_seed, axis=0)
        cost_real_ens = cost_ensembled[0]
        cost_random_ens = cost_ensembled[1:]
        ensembled_percentiles.append(
            100.0 * (cost_random_ens < cost_real_ens).sum() / len(cost_random_ens)
        )

    single_pass_percentiles = np.array(single_pass_percentiles)
    ensembled_percentiles = np.array(ensembled_percentiles)

    def report(name, percentiles):
        print(f'\n=== {name} ===')
        print(f'Mean percentile:   {percentiles.mean():.1f}')
        print(f'Median percentile: {np.median(percentiles):.1f}')
        print(f'Std percentile:    {percentiles.std():.1f}')
        print(f'% episodes real beats median random:  {100 * (percentiles < 50).mean():.1f}%')
        print(f'% episodes real in top decile (<10):  {100 * (percentiles < 10).mean():.1f}%')
        print(f'% episodes real in bottom decile (>90): {100 * (percentiles > 90).mean():.1f}%')

    print(f'\nN episodes: {len(idxs)}, K random candidates: {N_RANDOM_CANDIDATES}, '
          f'ensemble size M: {len(ENSEMBLE_SEEDS)} (seeds {ENSEMBLE_SEEDS})')
    report('Single-pass baseline (seed=0 only)', single_pass_percentiles)
    report(f'Ensembled (mean cost over {len(ENSEMBLE_SEEDS)} seeds)', ensembled_percentiles)

    print('\n=== Paired comparison (same 200 episodes, single-pass vs. ensembled) ===')
    diff = ensembled_percentiles - single_pass_percentiles
    corr = np.corrcoef(single_pass_percentiles, ensembled_percentiles)[0, 1]
    print(f'Pearson correlation (single-pass vs. ensembled percentile): {corr:.3f}')
    print(f'Mean per-episode change (ensembled - single): {diff.mean():+.1f}')
    print(f'Std of per-episode change:                    {diff.std():.1f}')
    flips_to_below_50 = ((single_pass_percentiles >= 50) & (ensembled_percentiles < 50)).sum()
    flips_to_above_50 = ((single_pass_percentiles < 50) & (ensembled_percentiles >= 50)).sum()
    print(f'Episodes flipping unreliable(>=50) -> reliable(<50): {flips_to_below_50}')
    print(f'Episodes flipping reliable(<50) -> unreliable(>=50): {flips_to_above_50}')


if __name__ == '__main__':
    main()
