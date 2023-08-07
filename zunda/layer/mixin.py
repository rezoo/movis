from typing import Sequence

import numpy as np


class TimelineMixin:
    def __init__(
        self, start_times: Sequence[float], end_times: Sequence[float]
    ) -> None:
        assert len(start_times) == len(
            end_times
        ), f"{len(start_times)} != {len(end_times)}"
        self.start_times: np.ndarray = np.asarray(start_times, dtype=float)
        self.end_times: np.ndarray = np.asarray(end_times, dtype=float)

    def get_state(self, time: float) -> int:
        idx = self.start_times.searchsorted(time, side="right") - 1
        if idx >= 0 and self.end_times[idx] > time:
            return int(idx)
        else:
            return -1

    @property
    def duration(self):
        return self.end_times[-1] - self.start_times[0]
