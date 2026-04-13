"""Progress bar utilities."""

from tqdm import tqdm

import torch.distributed as dist


class TqdmBar:
    """Progress bar wrapper for distributed training."""

    def __init__(
        self, 
        total: int, 
        desc: str, 
        unit: str, 
        colour: str = "blue", 
        rank: int = 0
    ) -> None:
        """
        Initialize the progress bar.
        
        Args:
            total: The total number of steps.
            desc: The description of the progress bar.
            unit: The unit of the progress bar.
            colour: The colour of the progress bar.
            rank: The rank of process to show the bar. If rank is "all", show for all ranks.
        """
        format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        if rank == "all" or dist.get_rank() == rank:
            self.pbar = tqdm(
                total=total, 
                desc=desc, 
                unit=unit, 
                bar_format=format, 
                colour=colour
            )
        else:
            self.pbar = None

    def update(self, step: int) -> None:
        if self.pbar is not None:
            self.pbar.update(step)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
