from typing import Hashable, Optional, Protocol, Sequence

import numpy as np

from zunda.attribute import Attribute


class Layer(Protocol):
    @property
    def duration(self):
        raise NotImplementedError

    def __call__(self, time: float) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        return time

    @property
    def attributes(self) -> dict[str, Attribute]:
        return dict()


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
