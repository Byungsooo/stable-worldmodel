import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.cjepa._videosaur.encoder import FrameEncoder
from stable_worldmodel.wm.cjepa._videosaur.groupers import SlotAttention
from stable_worldmodel.wm.cjepa._videosaur.initializer import RandomInit
from stable_worldmodel.wm.cjepa._videosaur.networks import MLP
from stable_worldmodel.wm.cjepa._videosaur.predictor import TransformerEncoder
from stable_worldmodel.wm.cjepa._videosaur.video import LatentProcessor, MapOverTime, ScanOverTime


class VideoSAUREncoder(nn.Module):
    """Frozen VideoSAUR slot encoder (DINOv2 -> 2-layer MLP -> recurrent Slot Attention).

    Reconstructs the exact architecture of the PushT VideoSAUR checkpoint
    released alongside C-JEPA (arXiv:2602.11389,
    huggingface.co/HazelNam/CJEPA, ``pusht_videosaur_model.ckpt``), so that
    checkpoint's ``state_dict`` loads via plain ``load_state_dict`` with no
    key remapping. See ``_videosaur/`` for the vendored (MIT-licensed,
    martius-lab/videosaur) building blocks this composes.

    Unlike ``DummySlotEncoder``, slot extraction is recurrent across time:
    Slot Attention runs frame-by-frame, carrying the (VideoSAUR-internal)
    predicted slot state from one frame into the next, with 3 extra
    correction iterations on the first frame only. So this module consumes
    and returns full ``(B, T, ...)`` clips rather than independent frames —
    signalled to ``CJEPAWorldModel._extract_slots`` via
    ``requires_temporal_context``.
    """

    requires_temporal_context = True

    def __init__(
        self,
        n_slots: int,
        slot_dim: int,
        img_size: int = 196,
        backbone_name: str = 'dinov2_small',
        backbone_feat_dim: int = 384,
        grouper_n_iters: int = 2,
        first_step_corrector_n_iters: int = 3,
        predictor_n_blocks: int = 1,
        predictor_n_heads: int = 4,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.img_size = img_size

        output_transform = MLP(
            backbone_feat_dim,
            slot_dim,
            [2 * backbone_feat_dim],
            initial_layer_norm=True,
            activation='relu',
        )
        frame_encoder = FrameEncoder(backbone_name=backbone_name, output_transform=output_transform)
        self.encoder = MapOverTime(frame_encoder)

        self.initializer = RandomInit(n_slots=n_slots, dim=slot_dim)

        grouper = SlotAttention(
            inp_dim=slot_dim, slot_dim=slot_dim, n_iters=grouper_n_iters, use_mlp=False
        )
        predictor = TransformerEncoder(
            dim=slot_dim, n_blocks=predictor_n_blocks, n_heads=predictor_n_heads
        )
        latent_processor = LatentProcessor(
            corrector=grouper,
            predictor=predictor,
            first_step_corrector_args={'n_iters': first_step_corrector_n_iters},
        )
        self.processor = ScanOverTime(latent_processor)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.requires_grad_(False)
        self.eval()

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f'VideoSAUREncoder checkpoint is missing expected keys: {missing}'
            )
        # `unexpected` is expected to be exactly the checkpoint's `decoder.*` /
        # loss/metric submodules, which this inference-only wrapper doesn't build.
        return unexpected

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, T, C, H, W) -> slots: (B, T, N, D)."""
        B, T, C, H, W = pixels.shape
        if (H, W) != (self.img_size, self.img_size):
            pixels = F.interpolate(
                pixels.flatten(0, 1),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False,
            ).unflatten(0, (B, T))

        encoder_out = self.encoder(pixels)
        features = encoder_out['features']  # (B, T, num_patches, slot_dim)

        init_slots = self.initializer(batch_size=B)  # (B, N, slot_dim)
        out = self.processor(init_slots, features)

        return out['state']  # (B, T, N, slot_dim)


__all__ = ['VideoSAUREncoder']
