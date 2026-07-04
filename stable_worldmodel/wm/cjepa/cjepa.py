import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DummySlotEncoder(nn.Module):
    """Placeholder slot encoder for CPU smoke-testing without VideoSAUR.

    Replace with VideoSAUREncoder (Phase 3) for real training.
    Input:  (B*T, C, H, W)
    Output: (B*T, N, D)
    """

    def __init__(
        self,
        n_slots: int,
        slot_dim: int,
        img_channels: int = 3,
        img_size: int = 64,
        checkpoint_path=None,
    ):
        # checkpoint_path is accepted (and ignored) so this class is a drop-in
        # swap for VideoSAUREncoder via the `model.slot_encoder._target_` CLI
        # override documented in cjepa.yaml, without needing to also strip
        # the sibling `checkpoint_path: null` key from the Hydra config.
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.proj = nn.Linear(img_channels * img_size * img_size, n_slots * slot_dim)
        self._img_size = img_size

    def forward(self, x):
        B = x.shape[0]
        return self.proj(x.view(B, -1)).view(B, self.n_slots, self.slot_dim)


def _hungarian_match_slots(pred, goal):
    """Align predicted slots to goal slots via Hungarian matching.

    Args:
        pred: (B, N, D)
        goal: (B, N, D)

    Returns:
        pred reordered to match goal, shape (B, N, D)
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return pred  # fallback: no matching

    B, N, D = pred.shape
    cost = torch.cdist(pred.float(), goal.float())  # (B, N, N)
    out = pred.clone()
    for b in range(B):
        _, col_ind = linear_sum_assignment(cost[b].detach().cpu().numpy())
        out[b] = pred[b, col_ind]
    return out


class CJEPAWorldModel(nn.Module):
    """Object-centric world model with causal inductive bias via object-level masking.

    Architecture (paper Sec 4.2):
      - Frozen slot encoder (VideoSAUR) extracts N object slots per frame.
      - During training: history slots for |M| randomly chosen objects are masked
        across time (except an identity anchor at t0), and all future slots are
        masked. The bidirectional predictor recovers all masked slots jointly.
      - At inference: no history masking; predictor receives full history + masked
        future and predicts one step forward (T_p = 1 for PushT).

    Implements the Costable protocol so it works with swm's CEM solver out of
    the box: get_cost -> rollout -> criterion.
    """

    def __init__(
        self,
        slot_encoder,
        predictor,
        action_encoder,
        n_slots: int,
        slot_dim: int,
        history_len: int,
        future_len: int,
        max_masked_slots: int,
        proprio_encoder=None,
    ):
        """
        Args:
            slot_encoder: Frozen encoder (B*T, C, H, W) -> (B*T, N, D).
                          Gradients are not computed through it.
            predictor:    BidirectionalTransformer operating on flattened tokens.
            action_encoder: Embedder mapping (B, T, act_dim) -> (B, T, slot_dim).
            n_slots:      N object slots per frame.
            slot_dim:     D, slot feature dimension (128 per paper).
            history_len:  T_h, number of context frames (3 for PushT).
            future_len:   T_p, number of predicted future frames (1 for PushT).
            max_masked_slots: Maximum |M| randomly masked objects during training.
            proprio_encoder: Optional Embedder for proprioception (B, T, p_dim) -> (B, T, D).
        """
        super().__init__()
        self.slot_encoder = slot_encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder

        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.history_len = history_len
        self.future_len = future_len
        self.total_len = history_len + future_len
        self.max_masked_slots = max_masked_slots

        # φ: linear projection for identity anchor (Eq. 3)
        self.anchor_proj = nn.Linear(slot_dim, slot_dim)

        # e_τ: learnable temporal positional embeddings (Eq. 3)
        from stable_worldmodel.wm.cjepa.module import TemporalPosEmb
        self.temporal_emb = TemporalPosEmb(self.total_len, slot_dim)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_slots(self, pixels):
        """Run frozen slot encoder on a batch of frames.

        Args:
            pixels: (B, T, C, H, W)

        Returns:
            slots: (B, T, N, D)
        """
        if getattr(self.slot_encoder, 'requires_temporal_context', False):
            # e.g. VideoSAUREncoder: Slot Attention is recurrent across time,
            # so the encoder needs the full clip, not independent frames.
            return self.slot_encoder(pixels)

        B, T = pixels.shape[:2]
        flat = rearrange(pixels, 'b t ... -> (b t) ...')
        slots = self.slot_encoder(flat)  # (B*T, N, D)
        return rearrange(slots, '(b t) n d -> b t n d', b=B)

    def encode(self, info):
        """Extract slots and auxiliary embeddings.

        Populates info with:
            'slots':    (B, T, N, D)  — target slots (all frames)
            'act_emb':  (B, T, 1, D)  — action aux node
            'prop_emb': (B, T, 1, D)  — proprio aux node (optional)
        """
        info['slots'] = self._extract_slots(
            info['pixels'].to(next(self.slot_encoder.parameters()).dtype)
        )

        if 'action' in info:
            act_emb = self.action_encoder(info['action'])  # (B, T, D)
            info['act_emb'] = act_emb.unsqueeze(2)  # (B, T, 1, D)

        if 'proprio' in info and self.proprio_encoder is not None:
            prop_emb = self.proprio_encoder(info['proprio'])  # (B, T, D)
            info['prop_emb'] = prop_emb.unsqueeze(2)  # (B, T, 1, D)

        return info

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _masked_token(self, anchor_slot, timestep_idx):
        """Construct masked token z̃_τ^i = φ(z_{t0}^i) + e_τ (Eq. 3).

        Args:
            anchor_slot: (*, D) — slot value at t0 (identity anchor)
            timestep_idx: int — τ
        """
        device = anchor_slot.device
        tau = torch.tensor(timestep_idx, device=device)
        return self.anchor_proj(anchor_slot) + self.temporal_emb(tau)

    def _build_masked_tokens(self, slots_all, act_emb, prop_emb=None):
        """Apply object-level masking and build the full predictor input.

        History masking (Sec 4.2): for each batch item, sample |M| ∈ {0,...,max_M}
        objects at random. For masked object i, replace slots at τ > 0 with the
        masked token φ(slot[t0]) + e_τ. The slot at t0 remains as an identity anchor.

        Future masking: all N objects at future timesteps are always masked.

        Args:
            slots_all: (B, T_total, N, D) — target slots for all frames
            act_emb:   (B, T_total, 1, D) or None
            prop_emb:  (B, T_total, 1, D) or None

        Returns:
            tokens:     (B, T_total * N_total, D) — masked input for predictor
            is_slot_masked: (B, T_total, N) bool — True where slot is a loss target
        """
        B, T, N, D = slots_all.shape
        T_h = self.history_len

        tokens = slots_all.clone()  # (B, T, N, D), will be modified in-place
        is_slot_masked = torch.zeros(B, T, N, dtype=torch.bool, device=slots_all.device)

        # Precompute temporal embeddings for all timesteps. Cast to tokens'
        # dtype: under mixed/low precision training, temporal_emb's output
        # dtype isn't guaranteed to match slots_all's, and the boolean-mask
        # assignment below (index_put) requires an exact dtype match, unlike
        # plain indexed assignment which casts implicitly.
        t_indices = torch.arange(T, device=slots_all.device)
        t_embs = self.temporal_emb(t_indices).to(tokens.dtype)  # (T, D)

        for b in range(B):
            # Sample how many objects to mask this batch item
            n_mask = torch.randint(0, self.max_masked_slots + 1, ()).item()
            masked_objs = torch.randperm(N)[:n_mask].tolist()

            # History masking: mask τ > t0 for chosen objects
            for i in masked_objs:
                anchor = slots_all[b, 0, i]  # identity anchor at t0
                for tau in range(1, T_h):
                    tokens[b, tau, i] = self.anchor_proj(anchor) + t_embs[tau]
                    is_slot_masked[b, tau, i] = True

            # Future masking: all objects at τ ≥ T_h
            for tau in range(T_h, T):
                for i in range(N):
                    anchor = slots_all[b, 0, i]
                    tokens[b, tau, i] = self.anchor_proj(anchor) + t_embs[tau]
                    is_slot_masked[b, tau, i] = True

        # Add temporal positional encoding to visible slots
        # (masked tokens already have it embedded; visible ones get it added here)
        visible_mask = ~is_slot_masked  # (B, T, N)
        tokens[visible_mask] = (
            tokens[visible_mask] + t_embs.unsqueeze(1).expand(B, T, N, D)[visible_mask]
        )

        # Concatenate auxiliary nodes (always visible, add temporal PE).
        # The future position(s) (tau >= T_h) must be conditioned on the
        # action block that actually PRODUCED that frame from the last
        # history frame (block T_h - 1), not the dataset's own block tau
        # (which is the action taken AFTER the target frame, uncorrelated
        # with how it was reached) — using block tau here was silently
        # teaching the model to associate the future/target token with an
        # action that has nothing to do with the frame it's predicting,
        # which matches this session's finding that forward_train's loss was
        # nearly insensitive to corrupting the action entirely. History
        # positions (tau < T_h) keep their own block tau ("action about to
        # be taken from here") as forward-looking context, matching what
        # rollout()'s current history-action handling assumes.
        aux_parts = []
        if act_emb is not None:
            act_emb_corrected = act_emb.clone()
            if T > T_h:
                act_emb_corrected[:, T_h:] = act_emb[:, T_h - 1 : T_h]
            aux_emb_t = act_emb_corrected + t_embs[:, None, :].unsqueeze(0)  # (B, T, 1, D)
            aux_parts.append(aux_emb_t)
        if prop_emb is not None:
            prop_emb_t = prop_emb + t_embs[:, None, :].unsqueeze(0)  # (B, T, 1, D)
            aux_parts.append(prop_emb_t)

        # Merge: (B, T, N + N_aux, D), then flatten to (B, T*(N+N_aux), D)
        all_entity_tokens = [tokens] + aux_parts  # list of (B, T, K, D)
        merged = torch.cat(all_entity_tokens, dim=2)  # (B, T, N_total, D)
        flat = rearrange(merged, 'b t n d -> b (t n) d')

        return flat, is_slot_masked

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train(self, batch):
        """Masked prediction training forward pass.

        Returns dict with 'loss', 'pred_loss'.
        """
        batch['action'] = torch.nan_to_num(batch.get('action', torch.zeros(1)), 0.0)

        info = self.encode(batch)
        slots_all = info['slots']  # (B, T, N, D) — targets

        act_emb = info.get('act_emb')
        prop_emb = info.get('prop_emb')

        tokens, is_masked = self._build_masked_tokens(slots_all, act_emb, prop_emb)
        # tokens: (B, T*(N+N_aux), D)
        # is_masked: (B, T, N) bool

        preds_flat = self.predictor(tokens)  # (B, T*(N+N_aux), D)

        # Extract slot predictions: first N entities per timestep
        N_total = self.n_slots + (1 if act_emb is not None else 0) + (
            1 if prop_emb is not None else 0
        )
        T = self.total_len
        preds = rearrange(preds_flat, 'b (t n) d -> b t n d', t=T, n=N_total)
        pred_slots = preds[:, :, : self.n_slots, :]  # (B, T, N, D)

        # MSE loss on all masked slot positions
        loss = F.mse_loss(pred_slots[is_masked], slots_all[is_masked].detach())

        return {'loss': loss, 'pred_loss': loss}

    # ------------------------------------------------------------------
    # Inference / MPC
    # ------------------------------------------------------------------

    def _single_step_predict(
        self, slots_hist, act_emb_step, hist_act_emb=None, prop_emb_step=None
    ):
        """Predict one future slot given history and next-step aux variables.

        Args:
            slots_hist:    (BS, T_h, N, D) — history slots (fully visible)
            act_emb_step:  (BS, 1, D) — action embedding for the future step.
                           Also reused, unshifted, as the LAST history
                           position's own forward action: block (T_h - 1) is
                           by definition "the action from the last history
                           frame to the new future frame" — exactly this same
                           candidate action, not a separate quantity.
            hist_act_emb:  (BS, T_h - 1, D) or None — real action embeddings
                           for the EARLIER history positions (0..T_h-2),
                           "the action about to be taken from here" (matches
                           _build_masked_tokens' forward-looking convention
                           for history). Falls back to zeros if not provided
                           (e.g. no action history is being tracked by the
                           caller).
            prop_emb_step: (BS, 1, D) or None

        Returns:
            pred_slots: (BS, N, D) — predicted future slot
        """
        BS, T_h, N, D = slots_hist.shape
        T_total = T_h + 1

        t_indices = torch.arange(T_total, device=slots_hist.device)
        t_embs = self.temporal_emb(t_indices)  # (T_total, D)

        # Build token sequence: history (visible) + 1 future (masked)
        # History: slots + temporal PE
        hist_tokens = slots_hist + t_embs[:T_h, None, :]  # (BS, T_h, N, D)

        # Future: masked token = anchor_proj(slot at t0) + e_T_h
        anchor = slots_hist[:, 0, :, :]  # (BS, N, D) — use t0 as anchor
        fut_tokens = self.anchor_proj(anchor) + t_embs[T_h]  # (BS, N, D)
        fut_tokens = fut_tokens.unsqueeze(1)  # (BS, 1, N, D)

        # Concatenate along time: (BS, T_total, N, D)
        slot_tokens = torch.cat([hist_tokens, fut_tokens], dim=1)

        # Aux nodes: history aux visible, future aux provided
        aux_parts = []
        if act_emb_step is not None:
            n_early = T_h - 1
            if hist_act_emb is not None and n_early > 0:
                early_hist_act = hist_act_emb + t_embs[:n_early].unsqueeze(0)  # (BS, n_early, D)
                early_hist_act = early_hist_act.unsqueeze(2)  # (BS, n_early, 1, D)
            else:
                early_hist_act = torch.zeros(
                    BS, n_early, 1, D, device=slot_tokens.device, dtype=slot_tokens.dtype
                )
            last_hist_act = (act_emb_step + t_embs[T_h - 1]).unsqueeze(1)  # (BS, 1, 1, D)
            fut_act = (act_emb_step + t_embs[T_h]).unsqueeze(1)  # (BS, 1, 1, D)
            act_tokens = torch.cat([early_hist_act, last_hist_act, fut_act], dim=1)  # (BS, T_total, 1, D)
            aux_parts.append(act_tokens)

        if prop_emb_step is not None:
            hist_prop = torch.zeros(BS, T_h, 1, D, device=slot_tokens.device, dtype=slot_tokens.dtype)
            fut_prop = prop_emb_step.unsqueeze(1) + t_embs[T_h]
            prop_tokens = torch.cat([hist_prop, fut_prop], dim=1)
            aux_parts.append(prop_tokens)

        all_tokens = torch.cat([slot_tokens] + aux_parts, dim=2)  # (BS, T_total, N_total, D)
        flat = rearrange(all_tokens, 'bs t n d -> bs (t n) d')

        preds_flat = self.predictor(flat)  # (BS, T_total * N_total, D)

        N_total = N + len(aux_parts)
        preds = rearrange(preds_flat, 'bs (t n) d -> bs t n d', t=T_total, n=N_total)
        pred_slots_future = preds[:, T_h, :N, :]  # (BS, N, D)
        return pred_slots_future

    def rollout(self, info, action_candidates):
        """Autoregressive latent rollout for MPC planning.

        Args:
            info: dict with:
                'pixels': (B, S, T_h, C, H, W) initial observation frames
                'goal':   (B, 1, 1, C, H, W) goal frame (or 'goal_slots' if pre-encoded)
            action_candidates: (B, S, T_plan, act_dim)

        Returns:
            info with added 'predicted_slots': (B, S, T_plan, N, D)
        """
        B, S, T_h = action_candidates.shape[:3]
        T_plan = action_candidates.shape[2]
        device = action_candidates.device

        # Encode initial history (once, shared across samples)
        if 'slots' not in info:
            pixels = info['pixels']  # (B, S, T_h, C, H, W)
            # encode per-sample: flatten B×S
            pixels_flat = rearrange(pixels, 'b s t ... -> (b s) t ...')
            slots_flat = self._extract_slots(
                pixels_flat.to(next(self.slot_encoder.parameters()).dtype)
            )  # (B*S, T_h, N, D)
            info['slots'] = rearrange(slots_flat, '(b s) t n d -> b s t n d', b=B, s=S)

        slots_hist = info['slots']  # (B, S, T_h, N, D)

        # Encode goal
        if 'goal_slots' not in info:
            goal_pixels = info['goal']  # (B, 1, 1, C, H, W) or (B, 1, C, H, W)
            if goal_pixels.dim() == 6:
                goal_pixels = goal_pixels[:, 0, 0]  # (B, C, H, W)
            elif goal_pixels.dim() == 5:
                goal_pixels = goal_pixels[:, 0]  # (B, C, H, W)
            goal_pixels = goal_pixels.unsqueeze(1)  # (B, 1, C, H, W)
            goal_slots = self._extract_slots(
                goal_pixels.to(next(self.slot_encoder.parameters()).dtype)
            )  # (B, 1, N, D)
            info['goal_slots'] = goal_slots[:, 0]  # (B, N, D)

        # Embed actions
        act_flat = rearrange(action_candidates, 'b s t d -> (b s) t d')
        act_emb_all = self.action_encoder(act_flat)  # (BS, T_plan, D)

        # Real observed history-action embeddings, if provided (see
        # CJEPAHistoryPolicy), for the early history positions (0..T_h-2).
        # This slides forward in lockstep with slots_h below: after each
        # rollout step, the window's "early" actions eventually become this
        # same loop's own earlier candidate actions rather than the original
        # real ones, exactly mirroring how slots_h mixes real and predicted
        # slots as it slides.
        act_hist = None
        if 'hist_action' in info and self.history_len > 1:
            hist_action = info['hist_action']  # (B, S, T_h-1, act_block_dim)
            hist_action_flat = rearrange(hist_action, 'b s t d -> (b s) t d')
            act_hist = self.action_encoder(
                hist_action_flat.to(act_emb_all.dtype)
            )  # (BS, T_h-1, D)

        predicted = []
        # Flatten B×S for rollout
        slots_h = rearrange(slots_hist, 'b s t n d -> (b s) t n d')

        for t in range(T_plan):
            act_step = act_emb_all[:, t, :]  # (BS, D)
            pred = self._single_step_predict(
                slots_h, act_step.unsqueeze(1), hist_act_emb=act_hist
            )  # (BS, N, D)
            predicted.append(pred)

            # Slide history window
            slots_h = torch.cat([slots_h[:, 1:], pred.unsqueeze(1)], dim=1)
            if act_hist is not None and act_hist.shape[1] > 0:
                act_hist = torch.cat([act_hist[:, 1:], act_step.unsqueeze(1)], dim=1)

        predicted_slots = torch.stack(predicted, dim=1)  # (BS, T_plan, N, D)
        info['predicted_slots'] = rearrange(
            predicted_slots, '(b s) t n d -> b s t n d', b=B, s=S
        )
        return info

    def criterion(self, info_dict):
        """L2 cost between last predicted slot and goal slot, with Hungarian matching.

        Returns:
            cost: (B, S)
        """
        pred = info_dict['predicted_slots']  # (B, S, T_plan, N, D)
        goal = info_dict['goal_slots']  # (B, N, D)

        pred_last = pred[:, :, -1, :, :]  # (B, S, N, D)
        B, S, N, D = pred_last.shape

        # Hungarian matching: align pred slots to goal slots per (b, s)
        pred_flat = rearrange(pred_last, 'b s n d -> (b s) n d')
        goal_flat = goal.unsqueeze(1).expand(B, S, N, D)
        goal_flat = rearrange(goal_flat, 'b s n d -> (b s) n d')

        pred_matched = _hungarian_match_slots(pred_flat, goal_flat)  # (BS, N, D)

        cost = ((pred_matched - goal_flat.detach()) ** 2).sum(dim=(-1, -2))  # (BS,)
        return rearrange(cost, '(b s) -> b s', b=B, s=S)

    def get_cost(self, info_dict, action_candidates):
        """Full MPC cost: rollout + Hungarian-matched L2 to goal.

        Implements the Costable protocol.
        """
        assert 'goal' in info_dict or 'goal_slots' in info_dict

        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


__all__ = ['CJEPAWorldModel']
