from __future__ import annotations
from typing import Callable, Hashable, Sequence

import numpy as np

from .layer import Layer


class Repeat:
    def __init__(self, layer: Layer, n_repeat: int = 1):
        assert 0 < n_repeat, f'n_loop must be positive integer, but {n_repeat}'
        self.layer = layer
        self.n_repeat = n_repeat

    @property
    def duration(self) -> float:
        return self.layer.duration * self.n_repeat

    def get_key(self, time: float) -> Hashable:
        if hasattr(self.layer, 'get_key'):
            return self.layer.get_key(time % self.layer.duration)
        return time % self.layer.duration

    def __call__(self, time: float) -> np.ndarray | None:
        return self.layer(time % self.layer.duration)


class TimeWarp:
    def __init__(self, layer: Layer, warp_func: Callable[[float], float], duration: float | None = None):
        self.layer = layer
        self.warp_func = warp_func
        self.duration = self.layer.duration if duration is None else duration

    def get_key(self, time: float) -> Hashable:
        return self.layer.get_key(self.warp_func(time))

    def __call__(self, time: float) -> np.ndarray | None:
        return self.layer(self.warp_func(time))


class Concatenate:
    def __init__(self, layers: Sequence[Layer]):
        self.layers = tuple(layers)
        self._timeline = np.cumsum([0.] + [layer.duration for layer in layers])

    @property
    def duration(self) -> float:
        return self._timeline[-1]

    def get_state(self, time: float) -> int:
        if time < 0. or self.duration <= time:
            return -1
        return int(np.searchsorted(self._timeline, time, side='right')) - 1

    def get_key(self, time: float) -> Hashable:
        idx = self.get_state(time)
        if idx < 0:
            return None
        layer = self.layers[idx]
        return layer.get_key(time - self._timeline[idx])

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0. or self.duration <= time:
            return None
        idx = np.searchsorted(self._timeline, time, side='right') - 1
        layer = self.layers[idx]
        return layer(time - self._timeline[idx])


class Trim:
    def __init__(self, layer: Layer, start_times: Sequence[float], end_times: Sequence[float]):
        assert len(start_times) == len(end_times)
        starts = np.array(start_times, dtype=np.float64)
        ends = np.array(end_times, dtype=np.float64)
        assert np.all(starts < ends)
        self.layer = layer
        self._durations = ends - starts
        self._duration = float(self._durations.sum())
        self._timeline = np.cumsum(np.concatenate([[0.], self._durations]))

    @property
    def duration(self) -> float:
        return self._duration

    def get_state(self, time: float) -> int:
        if time < 0. or self.duration <= time:
            return -1
        return int(np.searchsorted(self._timeline, time, side='right')) - 1

    def get_key(self, time: float) -> Hashable:
        idx = self.get_state(time)
        if idx < 0:
            return None
        return self.layer.get_key(time - self._timeline[idx])

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0. or self.duration <= time:
            return None
        idx = self.get_state(time)
        return self.layer(time - self._timeline[idx])
