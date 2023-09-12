from __future__ import annotations
from typing import Sequence

import numpy as np

from movis.layer.protocol import BasicLayer
from movis.layer.composition import Composition


def concatenate(layers: Sequence[BasicLayer], size: tuple[int, int] | None) -> Composition:
    """Concatenate layers into a single composition.

    Args:
        layers: Layers to concatenate.
        size: Size of the composition. If None, the size of the layer is estimated.

    Returns:
        Composition with all layers concatenated.
    """
    if size is None:
        shape = layers[0](0.0)
        if shape is None:
            raise ValueError("Cannot determine size of composition.")
        size = shape[1], shape[0]
    duration = sum(layer.duration for layer in layers)
    composition = Composition(size=size, duration=duration)
    time = 0.0
    for layer in layers:
        composition.add_layer(layer, offset=time)
        time += layer.duration
    return composition


def repeat(layer: BasicLayer, n_repeat: int, size: tuple[int, int] | None) -> Composition:
    """Repeat a layer multiple times.

    Args:
        layer:
            Layer to repeat.
        n_repeat:
            Number of times to repeat the layer.
        size:
            Size of the composition. If None, the size of the layer is estimated.

    Returns:
        Composition with the layer repeated.
    """
    if size is None:
        shape = layer(0.0)
        if shape is None:
            raise ValueError("Cannot determine size of composition.")
        size = shape[1], shape[0]
    duration = layer.duration * n_repeat
    composition = Composition(size=size, duration=duration)
    for i in range(n_repeat):
        composition.add_layer(layer, offset=i * layer.duration)
    return composition


def trim(
    layer: BasicLayer, start_times: Sequence[float], end_times: Sequence[float],
    size: tuple[int, int] | None
) -> Composition:
    """Trim a layer with given time intervals and concatenate them.

    Args:
        layer:
            Layer to trim.
        start_times:
            Start times of the intervals.
        end_times:
            End times of the intervals.
        size:
            Size of the composition. If None, the size of the layer is estimated."""
    assert 0 < len(start_times)
    assert len(start_times) == len(end_times)
    starts = np.array(start_times, dtype=np.float64)
    ends = np.array(end_times, dtype=np.float64)
    assert np.all(starts < ends)
    if size is None:
        shape = layer(0.0)
        if shape is None:
            raise ValueError("Cannot determine size of composition.")
        size = shape[1], shape[0]
    durations = ends - starts
    total_duration = float(durations.sum())
    offsets = np.cumsum(np.concatenate([[0.], durations]))[:-1] - starts

    composition = Composition(size=size, duration=total_duration)
    for start, end, offset in zip(starts, ends, offsets):
        composition.add_layer(layer, start_time=start, end_time=end, offset=offset)
    return composition
