from typing import Callable, Hashable, Optional, Sequence

import numpy as np

from .composition import Composition
from .layer import Layer


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


def concatenate(
        compositions: Sequence[Composition],
        size: Optional[tuple[int, int]] = None,
        duration: Optional[float] = None,
        names: Optional[Sequence[str]] = None) -> Composition:
    if size is None:
        size = compositions[0].size
    if duration is None:
        duration = sum([c.duration for c in compositions])
    if names is None:
        names = [f"scene_{i}" for i in range(len(compositions))]
    else:
        assert len(names) == len(compositions)

    main = Composition(size=size, duration=duration)
    offsets = np.cumsum([0] + [c.duration for c in compositions])
    for composition, name, offset in zip(compositions, names, offsets):
        main.add_layer(composition, name=name, offset=offset)
    return main


def trim(
        layer: Layer, start_times: Sequence[float], end_times: Sequence[float],
        size: tuple[int, int]) -> Composition:
    assert len(start_times) == len(end_times)
    starts = np.array(start_times, dtype=np.float64)
    ends = np.array(end_times, dtype=np.float64)
    assert np.all(starts < ends)
    durations = ends - starts
    offsets = np.cumsum(np.concatenate([[0.], durations]))
    composition = Composition(size=size, duration=offsets[-1])
    for start, end, offset, in zip(starts, ends, offsets):
        composition.add_layer(layer, offset=offset, start_time=start, end_time=end)
    return composition
