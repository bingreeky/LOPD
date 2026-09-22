from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def tensorboard_writer(log_dir: str | None):
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard is not installed; skipping TensorBoard logging")
        return None
    return SummaryWriter(log_dir=log_dir)
