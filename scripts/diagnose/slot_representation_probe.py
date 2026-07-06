"""Probe the frozen VideoSAUR slot encoder's representation quality directly,
independent of the CJEPA predictor/training. Two checks:

1. Content probe: does concatenated slot content linearly predict ground-truth
   object pose (block_pose, pos_agent)? A healthy frozen encoder should give a
   reasonable Ridge R^2 -- if this is near zero, the predictor has nothing to
   work with regardless of how well it's trained.

2. Identity/temporal stability: Hungarian-matched cosine similarity between
   consecutive frames' slots, at three different frame strides:
     - stride 1 (finest, raw env steps)
     - stride 2 (VideoSAUR's own training frameskip)
     - stride 5 (this project's actual data pipeline frameskip)
   If stability drops sharply from stride 2 to stride 5, that's direct
   evidence for the frameskip domain-shift risk flagged (but never checked)
   back in Phase 3 of CJEPA_PROJECT.md.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.wm.cjepa.videosaur_encoder import VideoSAUREncoder
from stable_worldmodel.wm.cjepa.download import download_videosaur_checkpoint
from stable_worldmodel.wm.cjepa.cjepa import _hungarian_match_slots

IMG_SIZE = 196
N_SLOTS = 4
SLOT_DIM = 128
N_EPISODES_PROBE = 50
FRAMES_PER_EPISODE = 20
PROBE_STRIDE = 5  # matches this project's training frameskip
N_EPISODES_STABILITY = 20
STABILITY_CLIP_LEN = 10
SEED = 1234


def build_encoder():
    ckpt_path = download_videosaur_checkpoint()
    encoder = VideoSAUREncoder(
        n_slots=N_SLOTS, slot_dim=SLOT_DIM, img_size=IMG_SIZE,
        checkpoint_path=str(ckpt_path),
    )
    return encoder.to('cuda').eval()


def preprocess(pixels_uint8):
    """(T, C, H, W) uint8 -> (1, T, C, H, W) float, ImageNet-normalized, resized."""
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source='pixels', target='pixels')
    resize = dt.transforms.Resize(IMG_SIZE, source='pixels', target='pixels')
    sample = {'pixels': pixels_uint8}
    sample = resize(to_image(sample))
    return sample['pixels'].unsqueeze(0)  # (1, T, C, H, W)


@torch.no_grad()
def content_probe(encoder, dataset, episode_ids):
    feats, block_pose, pos_agent = [], [], []
    for ep in episode_ids:
        ep_data = dataset.load_episode(int(ep))
        length = ep_data['pixels'].shape[0]
        frame_idxs = np.arange(0, length, PROBE_STRIDE)[:FRAMES_PER_EPISODE]
        if len(frame_idxs) < 2:
            continue
        clip = ep_data['pixels'][frame_idxs]  # (T, C, H, W) uint8
        clip = preprocess(clip).to('cuda')
        slots = encoder(clip)[0].float().cpu().numpy()  # (T, N, D)
        feats.append(slots.reshape(len(frame_idxs), -1))
        block_pose.append(ep_data['block_pose'][frame_idxs].numpy())
        pos_agent.append(ep_data['pos_agent'][frame_idxs].numpy())

    X = np.concatenate(feats, axis=0)
    Y_block = np.concatenate(block_pose, axis=0)
    Y_agent = np.concatenate(pos_agent, axis=0)

    results = {}
    for name, Y in [('block_pose', Y_block), ('pos_agent', Y_agent)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=SEED
        )
        reg = Ridge(alpha=10.0).fit(X_train, y_train)
        pred = reg.predict(X_test)
        results[name] = r2_score(y_test, pred, multioutput='variance_weighted')
    return results, X.shape[0]


@torch.no_grad()
def stability_check(encoder, dataset, episode_ids, stride):
    sims = []
    for ep in episode_ids:
        ep_data = dataset.load_episode(int(ep))
        length = ep_data['pixels'].shape[0]
        max_start = length - stride * STABILITY_CLIP_LEN
        if max_start <= 0:
            continue
        frame_idxs = np.arange(0, stride * STABILITY_CLIP_LEN, stride)
        clip = ep_data['pixels'][frame_idxs]
        clip = preprocess(clip).to('cuda')
        slots = encoder(clip)[0].float()  # (T, N, D)

        for t in range(slots.shape[0] - 1):
            a, b = slots[t : t + 1], slots[t + 1 : t + 2]  # (1, N, D)
            b_matched = _hungarian_match_slots(b, a)[0]  # (N, D)
            a0 = a[0]
            cos = torch.nn.functional.cosine_similarity(a0, b_matched, dim=-1)  # (N,)
            sims.append(cos.mean().item())
    return float(np.mean(sims)), float(np.std(sims)), len(sims)


def main():
    dataset = swm.data.load_dataset(
        'pusht_expert_train.lance', keys_to_cache=['action', 'proprio', 'state']
    )
    n_episodes = len(np.unique(dataset.get_col_data('episode_idx')))
    print(f'Dataset episodes available: {n_episodes}')

    encoder = build_encoder()
    rng = np.random.default_rng(SEED)

    print('\n=== Content probe: linear-probe slot features -> ground-truth pose ===')
    probe_eps = rng.choice(n_episodes, size=N_EPISODES_PROBE, replace=False)
    results, n_samples = content_probe(encoder, dataset, probe_eps)
    print(f'N samples: {n_samples} (from {N_EPISODES_PROBE} episodes, stride={PROBE_STRIDE})')
    for name, r2 in results.items():
        print(f'  R^2 vs {name}: {r2:.3f}')

    print('\n=== Identity/temporal stability: Hungarian-matched cosine similarity ===')
    stab_eps = rng.choice(n_episodes, size=N_EPISODES_STABILITY, replace=False)
    for stride, label in [(1, 'raw env step'), (2, "VideoSAUR's own training frameskip"), (5, 'this project data frameskip')]:
        mean_sim, std_sim, n = stability_check(encoder, dataset, stab_eps, stride)
        print(f'  stride={stride} ({label}): mean cos-sim={mean_sim:.3f} (std={std_sim:.3f}, n={n} frame-pairs)')


if __name__ == '__main__':
    main()
