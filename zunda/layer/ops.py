from typing import Hashable

from zunda.layer.layer import Layer


class Loop:

    def __init__(self, layer: Layer, n_loop: int = 1):
        assert 0 < n_loop, f'n_loop must be positive integer, but {n_loop}'
        self.layer = layer
        self.n_loop = n_loop

    @property
    def duration(self):
        return self.layer.duration * self.n_loop

    def get_key(self, time: float) -> Hashable:
        if hasattr(self.layer, 'get_key'):
            return self.layer.get_key(time % self.layer.duration)
        return time % self.layer.duration

    def __call__(self, time: float):
        return self.layer(time % self.layer.duration)
