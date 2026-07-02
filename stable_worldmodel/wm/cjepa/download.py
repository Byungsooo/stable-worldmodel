from pathlib import Path

from loguru import logger as logging

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.utils import HF_BASE_URL
from stable_worldmodel.wm.utils import _download


_REPO_ID = 'HazelNam/CJEPA'


def download_videosaur_checkpoint(cache_dir: str | None = None) -> Path:
    """Download the official PushT VideoSAUR checkpoint (~139MB) from HF.

    Cached under ``<cache_dir>/checkpoints/videosaur/pusht_videosaur_model.ckpt``.
    """
    dest_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / 'videosaur'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / 'pusht_videosaur_model.ckpt'

    if dest.exists():
        logging.info(f'VideoSAUR checkpoint already cached at {dest}')
        return dest

    url = f'{HF_BASE_URL}/{_REPO_ID}/resolve/main/pusht_videosaur_model.ckpt'
    logging.info(f'Downloading VideoSAUR PushT checkpoint from {url}')
    _download(url, dest)
    return dest


def download_videosaur_reference_slots(cache_dir: str | None = None) -> Path:
    """Download the authors' pre-extracted PushT slots (~4.8GB) for validation only.

    This is a one-time cross-check artifact for the Phase 3 smoke test, not a
    training input — callers should delete it after use (this Pod has no
    network volume and a ~20GB container disk).
    """
    dest_dir = get_cache_dir(cache_dir, sub_folder='tmp')
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / 'pusht_videosaur_slots.pkl'

    if dest.exists():
        logging.info(f'Reference slots already cached at {dest}')
        return dest

    url = f'{HF_BASE_URL}/{_REPO_ID}/resolve/main/pusht_videosaur_slots.pkl'
    logging.info(f'Downloading reference PushT slots (~4.8GB) from {url}')
    _download(url, dest)
    return dest


__all__ = ['download_videosaur_checkpoint', 'download_videosaur_reference_slots']
