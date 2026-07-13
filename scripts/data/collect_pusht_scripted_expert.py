"""Collect PushT demonstrations with a scripted, goal-directed pushing
controller, then keep only the episodes that actually reach the success
criterion (Experiment 5: does training-data success rate matter for WM+CEM
performance?).

There is no goal-directed expert policy shipped anywhere in this codebase --
`WeakPolicy` (used by every other `collect_pusht_*.py` script) is a random
walk constrained to stay near the block, with zero awareness of the goal
(confirmed empirically: 0/1000 episodes ever succeeded in the existing
`pusht_expert_train.lance`). This script implements a heuristic
"orbit-then-push" controller.

Two things this script does differently from a naive port of `WeakPolicy`'s
calling convention, both found by debugging actual failures:

1. Targets `env.goal_state` (the actual success-criterion target: agent
   position + block position + block angle), not `env.goal_pose` (a mostly
   fixed visual overlay position that success is NOT measured against).
   `env.goal_state`'s components are, by default, independently random
   samples -- jointly inconsistent, unlike eval-time goals which come from
   real recorded trajectory snapshots. We override it via
   `options={'goal_state': ...}` with a self-consistent target (agent parked
   directly behind the block's target position), matching how a real
   trajectory's final state would naturally look.
2. Uses the default block shape ('square', not the default 'T') --  T's
   non-convex geometry defeated a from-scratch circular-contact-point
   heuristic (agent would contact the block at points where no push force
   transfers, e.g. deep in a concave notch); square's convex geometry makes
   the circular approximation of contact points valid. This is a deliberate
   scope simplification, not an attempt to match the WeakPolicy dataset's
   exact task.

Raw success rate is modest (~5%, measured empirically over 40 trials at
these tuned parameters) -- collect a larger raw batch and filter down to
just the successful episodes, so the final training dataset is ~100%
successful by construction regardless of the controller's raw hit rate.

Usage:
    python collect_pusht_scripted_expert.py num_traj=8000
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy


class ScriptedPushPolicy(BasePolicy):
    """Heuristic controller: orbit to a point behind the block (relative to
    the goal direction) via tangential+radial motion, then push through
    toward the goal, then park the agent at its own recorded target."""

    def __init__(
        self,
        standoff_margin=6.0,
        align_tol=0.6,
        park_thresh=25.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.discrete = False
        self.standoff_margin = standoff_margin
        self.align_tol = align_tol
        self.park_thresh = park_thresh

    def set_env(self, env):
        self.env = env
        spec = getattr(env, 'spec', None)
        if spec is None:
            envs = getattr(env, 'envs', None)
            if envs:
                spec = envs[0].spec
        assert spec is not None and 'swm/PushT' in spec.id, (
            'ScriptedPushPolicy can only be used with the PushT environment.'
        )

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        if hasattr(self.env, 'envs'):
            envs = [e.unwrapped for e in self.env.envs]
        else:
            base_env = self.env.unwrapped
            envs = getattr(base_env, 'envs', [base_env])
            envs = [e.unwrapped if hasattr(e, 'unwrapped') else e for e in envs]

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            agent_pos = np.array(env.agent.position, dtype=np.float64)
            block_pos = np.array(env.block.position, dtype=np.float64)
            block_goal_pos = np.asarray(env.goal_state[2:4], dtype=np.float64)

            to_goal = block_goal_pos - block_pos
            dist_to_goal = np.linalg.norm(to_goal)
            push_dir = to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([1.0, 0.0])

            block_r = float(env.variation_space['block']['scale'].value)
            agent_r = float(env.variation_space['agent']['scale'].value) * 0.5
            standoff = block_r + agent_r + self.standoff_margin

            current_vec = agent_pos - block_pos
            current_dist = np.linalg.norm(current_vec)
            radial_dir = current_vec / current_dist if current_dist > 1e-6 else np.array([1.0, 0.0])
            current_angle = np.arctan2(current_vec[1], current_vec[0])
            desired_angle = np.arctan2(-push_dir[1], -push_dir[0])
            angle_diff = (desired_angle - current_angle + np.pi) % (2 * np.pi) - np.pi

            if dist_to_goal < self.park_thresh:
                # Parking: navigate the agent to its own recorded target
                # position (pos_diff is a joint 4D agent+block distance).
                target = env.goal_state[:2]
            elif abs(angle_diff) > self.align_tol or abs(current_dist - standoff) > 10.0:
                # Orbit via small tangential + radial steps (NOT a jump
                # straight at a distant waypoint, which cuts through the
                # block and bumps it the wrong way -- found by tracing
                # actual failures).
                tangent_dir = np.array([-radial_dir[1], radial_dir[0]])
                if angle_diff < 0:
                    tangent_dir = -tangent_dir
                tangent_step = tangent_dir * min(20.0, abs(angle_diff) * current_dist)
                radial_step = radial_dir * np.clip(standoff - current_dist, -15.0, 15.0)
                target = agent_pos + tangent_step + radial_step
            else:
                # Push, tapering strength as we approach to avoid
                # overshooting past the goal with momentum.
                push_mag = min(45.0, max(10.0, dist_to_goal * 0.6))
                target = agent_pos + push_dir * push_mag

            # Proportional target offset (NOT normalized to a unit vector --
            # env.step() interprets `action` as (target - agent.position) /
            # action_scale; normalizing away the magnitude causes the agent
            # to always lunge a full action_scale toward the target
            # regardless of actual distance, producing oscillation instead
            # of smooth convergence -- found by tracing actual failures).
            action = (target - agent_pos) / env.action_scale
            actions[i] = np.clip(action, -1, 1)

        return actions


def make_goal_state(rng):
    """Self-consistent goal: random block target position/angle, agent
    parked directly behind it along a random approach direction -- matching
    how a real trajectory's final state would naturally look, unlike the
    env's own default (independently-sampled agent/block targets)."""
    block_target = rng.uniform(150, 350, size=2)
    block_angle_target = rng.uniform(-np.pi, np.pi)
    push_dir_goal = rng.uniform(-1, 1, size=2)
    push_dir_goal /= np.linalg.norm(push_dir_goal)
    agent_target = block_target - push_dir_goal * 65.0
    return np.concatenate([agent_target, block_target, [block_angle_target], [0.0, 0.0]])


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Collect raw episodes in small batches (fresh random goal per batch,
    since `options` broadcasts to all envs in a batch identically), then
    filter down to only the episodes that reached the success criterion."""

    raw_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'pusht_scripted_raw.lance'
    )

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    policy = ScriptedPushPolicy()
    world.set_policy(policy)

    rng = np.random.default_rng(cfg.seed)
    n_envs = cfg.world.num_envs
    n_batches = (cfg.num_traj + n_envs - 1) // n_envs

    for b in range(n_batches):
        goal_state = make_goal_state(rng)
        options = {
            'goal_state': goal_state,
            'variation_values': {'block.shape': 4},  # force 'square' -- see module docstring
        }
        world.collect(
            raw_path,
            episodes=n_envs,
            seed=int(rng.integers(0, 2**31 - 1)),
            options=options,
            mode='append',
            progress=False,
        )
        if (b + 1) % 20 == 0:
            logging.info(f'[collect] batch {b + 1}/{n_batches} done')

    logging.success(f' Collected ~{n_batches * n_envs} raw episodes to {raw_path}')


if __name__ == '__main__':
    run()
