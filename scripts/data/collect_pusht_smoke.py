from pathlib import Path

import hydra
import stable_worldmodel as swm
from loguru import logger as logging

from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Collect a small disposable dataset for smoke-testing C-JEPA training,
    without pulling forward Phase 5's full pusht_expert_train collection."""

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(WeakPolicy(dist_constraint=100))

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'pusht_smoke.lance',
        episodes=50,
        seed=cfg.seed,
    )

    logging.success(' 🎉🎉🎉 Completed smoke-test data collection for pusht_smoke 🎉🎉🎉')


if __name__ == '__main__':
    run()
