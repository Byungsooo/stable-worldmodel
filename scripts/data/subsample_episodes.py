"""Write a new Lance dataset containing only the first N episodes of a
source dataset (by episode index), for cheap data-scale ablations without
recollecting.

Usage:
    python subsample_episodes.py <src.lance> <dst.lance> <n_episodes>
"""

import io
import sys

import numpy as np
from loguru import logger as logging
from PIL import Image

import stable_worldmodel as swm
from stable_worldmodel.data.format import get_format


def _decode_jpeg_stack(raw_bytes_list):
    return np.stack(
        [np.asarray(Image.open(io.BytesIO(b)).convert('RGB')) for b in raw_bytes_list]
    )


def main(src, dst, n_episodes):
    dataset = swm.data.load_dataset(src, keys_to_cache=[])
    col_name = 'episode_idx' if 'episode_idx' in dataset._schema_names else 'ep_idx'
    episode_idx = dataset.get_col_data(col_name).reshape(-1)
    step_idx = dataset.get_col_data('step_idx').reshape(-1)

    n_episodes = min(n_episodes, len(dataset.lengths))
    logging.info(f'Writing first {n_episodes}/{len(dataset.lengths)} episodes to {dst}')

    with get_format('lance').open_writer(dst, mode='overwrite') as writer:
        for ep in range(n_episodes):
            row_idx = np.nonzero(episode_idx == ep)[0]
            row_idx = row_idx[np.argsort(step_idx[row_idx])]
            ep_data = dataset.get_row_data(row_idx.tolist())
            ep_data.pop('id', None)
            for col in dataset.image_columns:
                if col in ep_data:
                    ep_data[col] = _decode_jpeg_stack(ep_data[col])
            writer.write_episode(ep_data)

    logging.success(f'Wrote {n_episodes} episodes to {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
