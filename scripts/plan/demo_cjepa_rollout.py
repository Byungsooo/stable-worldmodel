"""Visual demo: CJEPA MPC rollout + planner (CEM) action selection on PushT.

Loads the Phase-4 checkpoint (DummySlotEncoder, 8 epochs, toy dataset —
see CJEPA_PROJECT.md), runs it through CEM-based MPC on a live PushT env,
and renders both the executed rollout (agent/dataset/goal panel video) and
the CEM cost-convergence curve for every replanning step.

Fidelity caveat: DummySlotEncoder is an untrained fixed random projection
of pixels, and the checkpoint was only trained 8 epochs on 50 toy episodes.
This demonstrates the planning pipeline mechanics end-to-end, not good
task-solving behavior.
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import sys
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.callbacks import BestCostRecorder, EliteCostRecorder

sys.path.insert(0, str(Path(__file__).parent))
from cjepa_policy import CJEPAHistoryPolicy  # noqa: E402

CHECKPOINT = 'cjepa/weights_epoch_8.pt'
DATASET_NAME = 'pusht_smoke.lance'
MODEL_IMG_SIZE = 196  # matches cjepa.yaml's img_size used for this checkpoint
VIDEO_IMG_SIZE = 224  # raw render resolution for nicer-looking video panels
EVAL_BUDGET = 30
GOAL_OFFSET = 25
HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
OUTPUT_DIR = Path(__file__).parent / 'outputs' / 'cjepa_demo'


def img_transform(size, dtype=torch.float32):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=size),
        ]
    )


def build_process(dataset):
    process = {}
    for col in ['action', 'proprio', 'state']:
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != 'action':
            process[f'goal_{col}'] = scaler
    return process


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint {CHECKPOINT!r}...')
    model = swm.wm.utils.load_pretrained(CHECKPOINT)
    model = model.to('cuda').eval()
    model.requires_grad_(False)

    dataset = swm.data.load_dataset(
        DATASET_NAME, keys_to_cache=['action', 'proprio', 'state']
    )
    assert dataset.lengths[0] > GOAL_OFFSET + EVAL_BUDGET, (
        f'episode 0 (len={dataset.lengths[0]}) too short for '
        f'goal_offset={GOAL_OFFSET} + eval_budget={EVAL_BUDGET}'
    )

    process = build_process(dataset)
    transform = {
        'pixels': img_transform(MODEL_IMG_SIZE),
        'goal': img_transform(MODEL_IMG_SIZE),
    }

    plan_config = swm.PlanConfig(
        horizon=HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
    )
    solver = CEMSolver(
        model=model,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device='cuda',
        seed=42,
        callbacks=[BestCostRecorder(), EliteCostRecorder()],
    )

    # CEMSolver.solve() calls cb.reset() at the start of every replan, so its
    # own .history only ever holds the *latest* replan's cost curve. Capture
    # each replan's outputs['callbacks'] snapshot before it's overwritten.
    replan_costs = []
    orig_solve = solver.solve

    def solve_and_record(info_dict, init_action=None):
        outputs = orig_solve(info_dict, init_action=init_action)
        if 'callbacks' in outputs:
            replan_costs.append(
                {k: [list(b) for b in v] for k, v in outputs['callbacks'].items()}
            )
        return outputs

    solver.solve = solve_and_record

    policy = CJEPAHistoryPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )

    world = swm.World(
        'swm/PushT-v1',
        num_envs=1,
        max_episode_steps=2 * EVAL_BUDGET,
        image_shape=(VIDEO_IMG_SIZE, VIDEO_IMG_SIZE),
        render_mode='rgb_array',
    )
    world.set_policy(policy)

    print('Running MPC rollout...')
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=[0],
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        episodes_idx=[0],
        callables=[
            {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
            {
                'method': '_set_goal_state',
                'args': {'goal_state': {'value': 'goal_state'}},
            },
        ],
        video=OUTPUT_DIR,
    )
    print('metrics:', metrics)
    print(f'{len(replan_costs)} replanning steps recorded')

    # -- cost-convergence plot -------------------------------------------------
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, snapshot in enumerate(replan_costs):
        best = snapshot['BestCostRecorder'][0]
        elite_mean = [d['mean'] for d in snapshot['EliteCostRecorder'][0]]
        iters = range(len(best))
        ax.plot(iters, best, color='C0', alpha=0.35 + 0.65 * i / max(1, len(replan_costs) - 1))
        ax.plot(iters, elite_mean, color='C1', linestyle='--', alpha=0.25)
    ax.plot([], [], color='C0', label='best candidate cost')
    ax.plot([], [], color='C1', linestyle='--', label='elite-mean cost')
    ax.set_xlabel('CEM iteration')
    ax.set_ylabel('cost (Hungarian-matched slot L2)')
    ax.set_title(f'CEM convergence across {len(replan_costs)} replanning steps')
    ax.legend()
    fig.tight_layout()
    plot_path = OUTPUT_DIR / 'cem_convergence.png'
    fig.savefig(plot_path, dpi=120)
    print(f'Saved {plot_path}')

    video_path = OUTPUT_DIR / 'env_0.mp4'
    print(f'Video: {video_path} (exists={video_path.exists()})')


if __name__ == '__main__':
    main()
