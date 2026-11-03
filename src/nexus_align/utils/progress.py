"""Progress display: tqdm bar wrapper for distributed runs."""

from tqdm import tqdm

import torch


class TqdmBar:
    """
    Distributed-aware tqdm wrapper.

    - Shows the progress bar only on the first GPU of each node.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
        colour: str = "blue",
        mininterval=1.0,
    ) -> None:
        fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        display = torch.cuda.current_device() == 0

        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            bar_format=fmt,
            colour=colour,
            mininterval=mininterval
        ) if display else None

    def update(self, step: int) -> None:
        if self.pbar is not None:
            self.pbar.update(step)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
