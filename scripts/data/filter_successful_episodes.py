"""Filter a raw Lance dataset down to only the episodes that reached the
PushT success criterion (``terminated`` fired at least once), writing a new,
clean, ~100%-successful dataset for Experiment 5.

Usage:
    python filter_successful_episodes.py <src.lance> <dst.lance>
"""

import io
import sys
from pathlib import Path

import numpy as np
from loguru import logger as logging
from PIL import Image

import stable_worldmodel as swm
from stable_worldmodel.data.format import get_format


def _decode_jpeg_stack(raw_bytes_list):
    return np.stack(
        [np.asarray(Image.open(io.BytesIO(b)).convert('RGB')) for b in raw_bytes_list]
    )


def main(src, dst):
    dataset = swm.data.load_dataset(src, keys_to_cache=[])
    n_episodes = len(dataset.lengths)
    terminated = dataset.get_col_data('terminated').reshape(-1)
    col_name = 'episode_idx' if 'episode_idx' in dataset._schema_names else 'ep_idx'
    episode_idx = dataset.get_col_data(col_name).reshape(-1)

    successful_eps = [
        ep for ep in range(n_episodes) if terminated[episode_idx == ep].any()
    ]
    logging.info(
        f'{len(successful_eps)}/{n_episodes} episodes succeeded '
        f'({len(successful_eps) / n_episodes * 100:.1f}%)'
    )

    with get_format('lance').open_writer(dst, mode='overwrite') as writer:
        for ep in successful_eps:
            row_idx = np.nonzero(episode_idx == ep)[0]
            step_order = np.argsort(
                dataset.get_col_data('step_idx').reshape(-1)[row_idx]
            )
            row_idx = row_idx[step_order]
            ep_data = dataset.get_row_data(row_idx.tolist())
            ep_data.pop('id', None)
            for col in dataset.image_columns:
                if col in ep_data:
                    ep_data[col] = _decode_jpeg_stack(ep_data[col])
            writer.write_episode(ep_data)

    logging.success(f'Wrote {len(successful_eps)} successful episodes to {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
