from pathlib import Path
import numpy as np
import torch
import logging
import os
import random
from typing import Optional
from pytorch_lightning.utilities import rank_zero_warn

log = logging.getLogger(__name__)


def seed_everything(seed: Optional[int] = None):

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        rank_zero_warn(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

def get_checkpoint_path(model_dir):
    if os.path.isdir(model_dir):
        """Return path of latest checkpoint found in the model directory."""
        chkpt =  str(list(Path(model_dir).glob('checkpoints/*'))[-1])
    else: chkpt = model_dir
    return chkpt