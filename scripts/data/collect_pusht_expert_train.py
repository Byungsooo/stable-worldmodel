from pathlib import Path

import hydra
import stable_worldmodel as swm
from loguru import logger as logging

from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Collect the real pusht_expert_train dataset for Phase 5 training,
    matching docs/envs/pusht.md's documented pusht_expert baseline
    (1000 episodes, Weak Expert policy)."""

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(WeakPolicy(dist_constraint=100))

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'pusht_expert_train.lance',
        episodes=1000,
        seed=cfg.seed,
    )

    logging.success(' 🎉🎉🎉 Completed pusht_expert_train data collection 🎉🎉🎉')


if __name__ == '__main__':
    run()
