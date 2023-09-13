from __future__ import annotations

from typing import Sequence

import numpy as np

from movis.layer.composition import Composition
from movis.layer.protocol import BasicLayer


def concatenate(layers: Sequence[BasicLayer], size: tuple[int, int] | None) -> Composition:
    """Concatenate layers into a single composition.

    Args:
        layers:
            Layers to concatenate.
        size:
            Size of the composition. If None, the size of the layer is estimated.

    Returns:
        Composition with all layers concatenated.

    Examples:
        >>> import movis as mv
        >>> layer1 = mv.layer.Image("image1.png", duration=1.0)
        >>> layer2 = mv.layer.Image("image2.png", duration=2.0)
        >>> composition = mv.concatenate([layer1, layer2])  # concatenate two layers
        >>> composition.duration
        3.0
    """
    if size is None:
        img = layers[0](0.0)
        if img is None:
            raise ValueError("Cannot determine size of composition.")
        size = img.shape[1], img.shape[0]
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

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Image("image.png", duration=1.0)
        >>> composition = mv.repeat(layer, 3)  # repeat 3 times
        >>> composition.duration
        3.0
    """
    if size is None:
        img = layer(0.0)
        if img is None:
            raise ValueError("Cannot determine size of composition.")
        size = img.shape[1], img.shape[0]
    duration = layer.duration * n_repeat
    composition = Composition(size=size, duration=duration)
    for i in range(n_repeat):
        composition.add_layer(layer, offset=i * layer.duration)
    return composition


def trim(
    layer: BasicLayer, start_times: Sequence[float], end_times: Sequence[float],
    size: tuple[int, int] | None = None
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
            Size of the composition. If None, the size of the layer is estimated.

    Returns:
        Composition with the layer trimmed and concatenated.

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Video("video.mp4")
        >>> composition = mv.trim(layer, [0.0, 2.0], [1.0, 3.0])  # trim 1 second from the beginning and end
        >>> composition.duration
        2.0
    """
    assert 0 < len(start_times)
    assert len(start_times) == len(end_times)
    starts = np.array(start_times, dtype=np.float64)
    ends = np.array(end_times, dtype=np.float64)
    assert np.all(starts < ends)
    if size is None:
        img = layer(0.0)
        if img is None:
            raise ValueError("Cannot determine size of composition.")
        size = img.shape[1], img.shape[0]
    durations = ends - starts
    total_duration = float(durations.sum())
    offsets = np.cumsum(np.concatenate([[0.], durations]))[:-1] - starts

    composition = Composition(size=size, duration=total_duration)
    for start, end, offset in zip(starts, ends, offsets):
        composition.add_layer(layer, offset=offset, start_time=start, end_time=end)
    return composition


def tile(layers: Sequence[BasicLayer], rows: int, cols: int) -> Composition:
    """Tile layers into a single composition.

    Args:
        layers: Layers to tile.
        size: Size of the composition. If None, the size of the layer is estimated.

    Returns:
        Composition with all layers tiled.
    """
    assert len(layers) == rows * cols, \
        f"Number of layers ({len(layers)}) must be equal to rows * cols ({rows * cols})."
    result = layers[0](0.0)
    if result is None:
        raise ValueError("Cannot determine size of composition.")
    w, h = result.shape[1], result.shape[0]

    W = cols * w
    H = rows * h
    duration = max(layer.duration for layer in layers)
    composition = Composition(size=(W, H), duration=duration)
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * w
            y = (i + 0.5) * h
            composition.add_layer(
                layers[i * cols + j], position=(x, y))
    return composition
