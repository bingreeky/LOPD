from __future__ import annotations

import random
from typing import Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(f"{seed}-{epoch}").shuffle(order)
    return order


def next_batch(
    items: Sequence[T],
    seed: int,
    epoch: int,
    position: int,
    batch_size: int,
) -> tuple[list[T], int, int]:
    target = min(batch_size, len(items))
    indices: list[int] = []
    while len(indices) < target:
        order = epoch_order(len(items), seed, epoch)
        take = min(target - len(indices), len(order) - position)
        indices.extend(order[position:position + take])
        position += take
        if position >= len(order):
            epoch, position = epoch + 1, 0
    return [items[i] for i in indices], epoch, position
