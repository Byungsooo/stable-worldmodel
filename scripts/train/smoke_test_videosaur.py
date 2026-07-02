"""Phase 3 smoke test for VideoSAUREncoder (C-JEPA project).

Downloads the official PushT VideoSAUR checkpoint, builds VideoSAUREncoder,
and runs it on a short clip of *real* PushT frames (rendered live from the
`swm/PushT-v1` gymnasium env — this Pod hasn't collected the full
`pusht_expert_train.lance` training dataset yet, which is a Phase 5 task,
so this avoids pulling that in early). Verifies output shape, absence of
NaNs, and that slots evolve across time (temporal recurrence is actually
doing something, not collapsing to a static per-clip value).

Does NOT attempt the numerical cross-check against the authors'
`pusht_videosaur_slots.pkl` (see download_videosaur_reference_slots) since
that requires the exact same dataset clips used at extraction time — left
for whoever picks up Phase 5's data collection.

Usage:
    python scripts/train/smoke_test_videosaur.py
"""

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401  (registers swm/PushT-v1)
import torch
import torch.nn.functional as F

from stable_worldmodel.wm.cjepa import VideoSAUREncoder
from stable_worldmodel.wm.cjepa.download import download_videosaur_checkpoint


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def collect_pusht_clip(n_frames: int, img_size: int) -> torch.Tensor:
    """Render a short clip of real PushT frames via the live simulator."""
    env = gym.make('swm/PushT-v1', render_mode='rgb_array')
    env.reset(seed=0)

    frames = []
    for _ in range(n_frames):
        frame = env.render()  # (H, W, 3) uint8
        frames.append(frame)
        action = env.action_space.sample()
        env.step(action)
    env.close()

    clip = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, C)
    clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)
    clip = F.interpolate(clip, size=(img_size, img_size), mode='bilinear', align_corners=False)
    clip = (clip.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD  # (1, T, C, H, W)
    return clip


def main():
    n_slots, slot_dim, img_size, n_frames = 4, 128, 196, 4

    checkpoint_path = download_videosaur_checkpoint()
    print(f'Checkpoint: {checkpoint_path}')

    encoder = VideoSAUREncoder(
        n_slots=n_slots, slot_dim=slot_dim, img_size=img_size, checkpoint_path=str(checkpoint_path)
    )

    pixels = collect_pusht_clip(n_frames, img_size)
    slots = encoder(pixels)

    print(f'slots shape: {tuple(slots.shape)}')
    assert slots.shape == (1, n_frames, n_slots, slot_dim), (
        f'Expected shape (1, {n_frames}, {n_slots}, {slot_dim}), got {tuple(slots.shape)}'
    )
    assert torch.isfinite(slots).all(), 'Slots contain NaN/Inf'

    frame_to_frame_delta = (slots[0, 1:] - slots[0, :-1]).abs().mean().item()
    print(f'mean |slots[t+1] - slots[t]|: {frame_to_frame_delta:.4f} (should be > 0)')
    assert frame_to_frame_delta > 1e-6, 'Slots are static across time — recurrence looks broken'

    print('Phase 3 smoke test PASSED: real shapes, finite values, temporal signal present.')


if __name__ == '__main__':
    main()
