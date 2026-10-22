"""Progress display: tqdm bar wrapper for distributed runs."""

from tqdm import tqdm

import torch
import torch.distributed as dist


class TqdmBar:
    """
    Distributed-aware tqdm wrapper.

    - Shows the progress bar only on the selected rank.
    - Uses rank="all" to show a bar on every rank.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
        colour: str = "blue",
        mininterval=1.0,
        display_on: int | str = "node",
    ) -> None:
        fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        display = False
        if display_on == "all":
            display = True
        if display_on == "node" and torch.cuda.current_device() == 0:
            display = True
        
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
