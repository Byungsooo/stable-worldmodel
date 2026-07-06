"""Slot initializer, vendored from martius-lab/videosaur (MIT License).

Source: videosaur/modules/initializers.py — ``RandomInit`` only, the variant
used by VideoSAUR's PushT config. Note initialization is stochastic (sampled
noise around a learned mean/std); only the very first frame of a clip uses
this — subsequent frames carry forward the previous frame's slot state via
``ScanOverTime`` (see video.py).
"""

from typing import Optional

import torch
from torch import nn


class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))
        # Settable so a caller can deliberately vary the slot-space anchor
        # across independent ensemble passes (see
        # scripts/diagnose/ensembled_action_sensitivity_check.py) while a
        # single get_cost() call still sees one consistent seed across its
        # internal history/goal encodings. Default 0 preserves the exact
        # behavior this was fixed to have.
        self.seed = 0

    def forward(self, batch_size: int):
        # Deterministic per-slot noise instead of a fresh torch.randn every
        # call: at CJEPA inference time, rollout() encodes history and goal
        # frames in separate _extract_slots() calls (unlike training, which
        # encodes a whole clip in one call via a shared draw), so unseeded
        # noise here made the same pixel content map to unrelated points in
        # slot-space depending purely on which call encoded it, swamping any
        # real signal in the downstream cost comparison.
        generator = torch.Generator(device=self.mean.device).manual_seed(self.seed)
        base_noise = torch.randn(
            self.n_slots, self.dim, device=self.mean.device, generator=generator
        )
        noise = base_noise.unsqueeze(0).expand(batch_size, -1, -1)
        return self.mean + noise * self.log_std.exp()
