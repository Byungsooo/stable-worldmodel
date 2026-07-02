"""Frame encoder, adapted from galilai-group/cjepa's VideoSAUR fork
(vendored subset of martius-lab/videosaur, MIT License, modified by the
C-JEPA authors to use a HuggingFace-loaded DINOv2 backbone instead of timm).

Source: src/third_party/videosaur/videosaur/modules/encoders.py in
https://github.com/galilai-group/cjepa (the ``pusht_dinov2_hf.yml`` config's
name suffix "_hf" refers to this HuggingFace-backbone variant). Confirmed by
inspecting the released ``pusht_videosaur_model.ckpt`` checkpoint directly:
its backbone parameter names (``embeddings.cls_token``,
``encoder.layer.0.attention.attention.key``, ``layernorm.weight``, ...) match
``transformers.Dinov2Model``, not timm's ``VisionTransformer`` — so this
project reuses ``facebook/dinov2-small`` via ``transformers.AutoModel``,
the same loading path already used elsewhere in this repo (see
``stable_worldmodel/wm/prejepa/module.py::create_backbone``), rather than
adding a `timm` dependency.

Trimmed to the single feature ("vit_block12", i.e. the backbone's final
post-layernorm patch tokens) actually consumed downstream — the reference
fork also derives a `vit_block_keys12` feature, but that is only used by a
VideoSAUR *training*-time auxiliary loss, not needed to extract slots.
"""

from typing import Dict, Optional

import torch
from torch import nn

from stable_worldmodel.wm.prejepa.module import create_backbone


class FrameEncoder(nn.Module):
    """Reduces a single frame to patch tokens (via frozen DINOv2) then to slot-space features."""

    def __init__(
        self,
        backbone_name: str = 'dinov2_small',
        output_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name)
        self.backbone.config.output_hidden_states = True
        self.output_transform = output_transform

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        with torch.no_grad():
            out = self.backbone(images, output_hidden_states=True)

        backbone_features = out.last_hidden_state.detach()[:, 1:, :]  # drop CLS token
        features = backbone_features.clone()

        if self.output_transform:
            features = self.output_transform(features)

        assert features.ndim == 3, (
            f'Expect output shape (batch, tokens, dims), but got {features.shape}'
        )

        return {'features': features, 'backbone_features': backbone_features}
