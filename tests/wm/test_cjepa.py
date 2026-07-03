"""Phase 4 smoke tests for CJEPAWorldModel.

Two kinds of coverage:
  1. get_cost shape-contract test, mirroring test_lewm.py / test_pldm.py's
     bare-model + monkey-patched-rollout/criterion style.
  2. Real forward_train/backward + masking-invariant tests against the actual
     BidirectionalTransformer/TemporalPosEmb/DummySlotEncoder modules, since
     no sibling wm test exercises a real forward+loss pass yet.
"""

import torch

from stable_worldmodel.wm.cjepa.cjepa import CJEPAWorldModel, DummySlotEncoder
from stable_worldmodel.wm.cjepa.module import BidirectionalTransformer, Embedder

# CEM-like dimensions for the get_cost shape test
B, S, T_plan, N, D = 2, 3, 2, 4, 5


def _bare_model():
    """Bypass CJEPAWorldModel.__init__; we only need get_cost to run."""
    return object.__new__(CJEPAWorldModel)


# ---------------------------------------------------------------------------
# get_cost shape contract: rollout -> criterion, per the Costable protocol
# ---------------------------------------------------------------------------


def test_cjepa_get_cost_wires_rollout_into_criterion():
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = {
        'pixels': torch.randn(B, S, 3, 3, 8, 8),
        'goal': torch.randn(B, 1, 1, 3, 8, 8),
    }
    action_candidates = torch.randn(B, S, T_plan, 2)

    rollout_call_count = []

    def mock_rollout(info, ac):
        rollout_call_count.append(1)
        assert ac.shape == (B, S, T_plan, 2), (
            f'rollout received wrong action_candidates shape: {ac.shape}'
        )
        info['predicted_slots'] = torch.randn(B, S, T_plan, N, D)
        return info

    def mock_criterion(info):
        assert info['predicted_slots'].shape == (B, S, T_plan, N, D), (
            f'criterion received wrong predicted_slots shape: '
            f'{info["predicted_slots"].shape}'
        )
        return torch.zeros(B, S)

    model.rollout = mock_rollout
    model.criterion = mock_criterion

    cost = CJEPAWorldModel.get_cost(model, info_dict, action_candidates)

    assert len(rollout_call_count) == 1, 'rollout should be called exactly once'
    assert cost.shape == (B, S), f'cost shape: expected ({B},{S}), got {cost.shape}'


def test_cjepa_get_cost_requires_goal():
    """get_cost asserts a goal (raw or pre-encoded) is present before rolling out."""
    model = _bare_model()
    model.rollout = lambda info, ac: info
    model.criterion = lambda info: torch.zeros(B, S)

    info_dict = {'pixels': torch.randn(B, S, 3, 3, 8, 8)}  # no 'goal'/'goal_slots'
    action_candidates = torch.randn(B, S, T_plan, 2)

    try:
        CJEPAWorldModel.get_cost(model, info_dict, action_candidates)
        raised = False
    except AssertionError:
        raised = True

    assert raised, "get_cost should assert when neither 'goal' nor 'goal_slots' is present"


# ---------------------------------------------------------------------------
# Real forward_train + backward smoke test
# ---------------------------------------------------------------------------

# Tiny real-module dimensions
TB, TT, TN, TD, TA = 2, 3, 2, 8, 2  # batch, total_len, n_slots, slot_dim, action_dim
HIST, FUT, MAX_MASKED = 2, 1, 1


def _make_tiny_model():
    return CJEPAWorldModel(
        slot_encoder=DummySlotEncoder(n_slots=TN, slot_dim=TD, img_size=8),
        predictor=BidirectionalTransformer(
            dim=TD, depth=1, heads=2, dim_head=4, mlp_dim=16
        ),
        action_encoder=Embedder(input_dim=TA, smoothed_dim=TD, emb_dim=TD),
        n_slots=TN,
        slot_dim=TD,
        history_len=HIST,
        future_len=FUT,
        max_masked_slots=MAX_MASKED,
    )


def _make_tiny_batch():
    return {
        'pixels': torch.randn(TB, TT, 3, 8, 8),
        'action': torch.randn(TB, TT, TA),
    }


def test_cjepa_forward_train_loss_is_finite_and_positive():
    torch.manual_seed(0)
    model = _make_tiny_model()
    batch = _make_tiny_batch()

    out = model.forward_train(batch)

    assert torch.isfinite(out['loss']).all(), 'loss is not finite (NaN/Inf)'
    assert out['loss'].item() > 0, 'loss should be positive for random init/targets'
    assert torch.equal(out['loss'], out['pred_loss'])


def test_cjepa_forward_train_backward_flows_gradients_to_predictor():
    torch.manual_seed(0)
    model = _make_tiny_model()
    batch = _make_tiny_batch()

    out = model.forward_train(batch)
    out['loss'].backward()

    grads = [p.grad for p in model.predictor.parameters()]
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads), (
        'expected at least one predictor parameter to receive a nonzero gradient'
    )


# ---------------------------------------------------------------------------
# Masking invariants: _build_masked_tokens
# ---------------------------------------------------------------------------


def test_cjepa_build_masked_tokens_shapes_and_invariants():
    torch.manual_seed(0)
    model = _make_tiny_model()

    slots_all = torch.randn(TB, TT, TN, TD)
    act_emb = torch.randn(TB, TT, 1, TD)

    tokens, is_slot_masked = model._build_masked_tokens(slots_all, act_emb)

    # is_slot_masked: (B, T, N) bool
    assert is_slot_masked.shape == (TB, TT, TN)
    assert is_slot_masked.dtype == torch.bool

    # future timesteps (tau >= history_len) are always fully masked
    assert is_slot_masked[:, HIST:, :].all(), 'all future slots must be masked'

    # t0 anchor is never masked
    assert not is_slot_masked[:, 0, :].any(), 't0 anchor slots must never be masked'

    # history masked-object count respects max_masked_slots
    history_masked_any_tau = is_slot_masked[:, 1:HIST, :].any(dim=1)  # (B, N)
    n_masked_per_batch = history_masked_any_tau.sum(dim=-1)  # (B,)
    assert (n_masked_per_batch >= 0).all() and (n_masked_per_batch <= MAX_MASKED).all(), (
        f'masked history object count out of bounds: {n_masked_per_batch.tolist()}'
    )

    # tokens: (B, T * (N + N_aux), D); N_aux = 1 for the action aux node
    n_total = TN + 1
    assert tokens.shape == (TB, TT * n_total, TD)


def test_cjepa_build_masked_tokens_history_mask_count_varies_with_seed():
    """Sanity check that masking is actually randomized, not a fixed pattern."""
    model = _make_tiny_model()
    slots_all = torch.randn(TB, TT, TN, TD)
    act_emb = torch.randn(TB, TT, 1, TD)

    counts = set()
    for seed in range(10):
        torch.manual_seed(seed)
        _, is_slot_masked = model._build_masked_tokens(slots_all, act_emb)
        history_masked_any_tau = is_slot_masked[:, 1:HIST, :].any(dim=1)
        counts.add(int(history_masked_any_tau.sum().item()))

    assert len(counts) > 1, 'expected masked-object count to vary across seeds'
