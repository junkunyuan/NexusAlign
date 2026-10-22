"""Dataset utils: generic helpers shared across datasets."""

import random
from typing import TypeVar

T = TypeVar("T")


def sample_items(
    items: list[T],
    sample_ratio: float | int | None = None,
    seed: int | None = None,
) -> list[T]:
    """
    Randomly subsample items.

    If sample_ratio<=0, keeps all.
    If sample_ratio<=1, sample the specific fraction.
    Else, sample the specific count.
    If seed is set, use a local deterministic RNG.
    """
    n = len(items)
    if not isinstance(sample_ratio, (int, float)) or sample_ratio <= 0 or n == 0:
        return items
    k = min(int(n * sample_ratio if sample_ratio < 1.1 else sample_ratio), n)
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(items, k)
