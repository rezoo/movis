from typing import Callable, Hashable, Optional

import numpy as np

from zunda.layer.layer import Layer


class Loop:

    def __init__(self, layer: Layer, n_loop: int = 1):
        assert 0 < n_loop, f'n_loop must be positive integer, but {n_loop}'
        self.layer = layer
        self.n_loop = n_loop

    @property
    def duration(self) -> float:
        return self.layer.duration * self.n_loop

    def get_key(self, time: float) -> Hashable:
        if hasattr(self.layer, 'get_key'):
            return self.layer.get_key(time % self.layer.duration)
        return time % self.layer.duration

    def __call__(self, time: float) -> Optional[np.ndarray]:
        return self.layer(time % self.layer.duration)


class TimeWarp:

    def __init__(self, layer: Layer, warp_func: Callable[[float], float], duration: Optional[float] = None):
        self.layer = layer
        self.warp_func = warp_func
        self.duration = self.layer.duration if duration is None else duration

    def get_key(self, time: float) -> Hashable:
        return self.layer.get_key(self.warp_func(time))

    def __call__(self, time: float) -> Optional[np.ndarray]:
        return self.layer(self.warp_func(time))
