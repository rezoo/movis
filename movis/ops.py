from __future__ import annotations

from typing import Sequence

import numpy as np

from movis.enum import Direction
from movis.layer.composition import Composition
from movis.layer.protocol import BasicLayer


def concatenate(layers: Sequence[BasicLayer], size: tuple[int, int] | None = None) -> Composition:
    """Concatenate layers into a single composition.

    Args:
        layers:
            Layers to concatenate.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.

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


def repeat(layer: BasicLayer, n_repeat: int, size: tuple[int, int] | None = None) -> Composition:
    """Repeat a layer multiple times.

    Args:
        layer:
            Layer to repeat.
        n_repeat:
            Number of times to repeat the layer.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.

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
            Size of the composition. If ``None``, the size of the layer is estimated.

    Returns:
        Composition with the layer trimmed and concatenated.

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Video("video.mp4")
        >>> composition = mv.trim(layer, [0.0, 2.0], [1.0, 3.0])  # select 0.0-1.0 and 2.0-3.0, and concatenate them
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


def tile(
    layers: Sequence[BasicLayer], rows: int, cols: int,
    size: tuple[int, int] | None = None,
) -> Composition:
    """Tile layers into a single composition.

    Args:
        layers:
            Layers to tile. Note that the order of the layers is row-major.
            For example, if ``layers`` is ``[a, b, c, d]`` and
            ``rows`` and ``cols`` are both ``2``, the composition will be: ``[[a, b], [c, d]]``.
        rows:
            Number of rows.
        cols:
            Number of columns.
        size:
            Size of each layer. Note that ``tile`` assumes that all layers have the same size.
            If ``None``, the size of the layer is estimated.

    .. note::
        The layer resolution specified in ``size`` does not have to be the actual layer resolution.
        For example, if ``size`` is specified to be larger than the actual layer size,
        each layer is placed in the center of each tile.

    Returns:
        Composition with all layers tiled. The size is ``(cols * w, rows * h)``,
        where ``w`` and ``h`` are the width and height of each layer, respectively.

    Examples:
        >>> import movis as mv
        >>> import numpy as np
        >>> layer1 = mv.layer.Image(np.zeros((100, 100, 4), dtype=np.uint8), duration=1.0)
        >>> layer2 = mv.layer.Image(np.zeros((100, 100, 4), dtype=np.uint8), duration=1.0)
        >>> composition = mv.tile([layer1, layer2], rows=1, cols=2)  # tile 1x2
        >>> composition.duration
        1.0
        >>> composition.size
        (200, 100)
    """
    assert len(layers) == rows * cols, \
        f"Number of layers ({len(layers)}) must be equal to rows * cols ({rows * cols})."
    if size is None:
        result = layers[0](0.0)
        if result is None:
            raise ValueError("Cannot determine size of composition.")
        w, h = result.shape[1], result.shape[0]
    else:
        w, h = size

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


def crop(layer: BasicLayer, rect: tuple[int, int, int, int]) -> Composition:
    """Crop a layer from a specified rectangle.

    Args:
        layer:
            Layer to crop.
        rect:
            Rectangle to crop. (x, y, width, height)

    Returns:
        Composition with the layer cropped.

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Image("image.png", duration=1.0)
        >>> composition = mv.crop(layer, (10, 20, 100, 200))  # crop from (10, 20) with size (100, 200)
        >>> composition.duration
        1.0
        >>> composition.size
        (100, 200)
    """
    assert len(rect) == 4
    x, y, w, h = rect
    duration = layer.duration
    composition = Composition(size=(w, h), duration=duration)
    composition.add_layer(
        layer, position=(-x, -y), origin_point=Direction.TOP_LEFT)
    return composition


def switch(
    layers: Sequence[BasicLayer],
    start_times: Sequence[float], cams: Sequence[int],
    size: tuple[int, int] | None = None, duration: float | None = None
) -> Composition:
    """Switch layers at specified times.

    Args:
        layers:
            Layers to switch.
        start_times:
            Start times of the intervals.
        cams:
            Scene numbers of the intervals.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.
        duration:
            Duration of the composition. If ``None``, the duration is estimated.

    Returns:
        Composition with the layer switched.

    Examples:
        >>> import movis as mv
        >>> layer1 = mv.layer.Image("image1.png", duration=5.0)
        >>> layer2 = mv.layer.Image("image2.png", duration=5.0)
        >>> # Show layer1 at 0.0-2.0 and layer2 at 2.0-5.0
        >>> composition = mv.switch([layer1, layer2], [0.0, 2.0], [0, 1])
        >>> composition.duration
        5.0
    """
    assert len(start_times) == len(cams)
    assert 0 <= min(cams)
    assert max(cams) < len(layers)
    starts = np.array(start_times, dtype=np.float64)
    assert np.all(starts[1:] > starts[:-1])
    if size is None:
        img = layers[0](0.0)
        if img is None:
            raise ValueError("Cannot determine size of composition.")
        size = img.shape[1], img.shape[0]
    if duration is None:
        duration = min(layer.duration for layer in layers)
    composition = Composition(size=size, duration=duration)
    times = np.concatenate([starts, [duration]])
    for start_time, end_time, ind in zip(times[:-1], times[1:], cams):
        composition.add_layer(
            layers[ind], start_time=start_time, end_time=end_time)
    return composition


def insert(
    source: BasicLayer, target: BasicLayer, time: float,
    size: tuple[int, int] | None = None
) -> Composition:
    """Insert a target layer into a source layer at a specified time.

    For instance, consider inserting a brief eye-catch scene to indicate a change in chapters within
    a long interview video. ``mv.insert()`` is used for such purposes, to insert a short scene into a longer one:

    |------------------------|    |------------|----------|------------|
    |    Source layer        | -> |   Source   |  target  |   Source   |
    |------------------------|    |------------|----------|------------|

    Args:
        source:
            The layer to insert the target layer into.
        target:
            The layer to insert.
        time:
            The time to insert the target layer.
        size:
            Size of the returned composition. If ``None``, the size of the source layer is used.

    Returns:
        Composition with the target layer inserted.

    Examples:
        >>> import movis as mv
        >>> source = mv.layer.Image("source.png", duration=5.0)
        >>> target = mv.layer.Image("target.png", duration=1.0)
        >>> composition = mv.insert(source, target, time=2.0)
        >>> composition.duration
        6.0
        >>> composition(0.0)  # source layer
        >>> composition(2.0)  # target layer
        >>> composition(3.0)  # source layer
    """
    if size is None:
        img = source(0.0)
        if img is None:
            raise ValueError("Cannot determine size of composition.")
        size = img.shape[1], img.shape[0]
    duration = source.duration + target.duration
    composition = Composition(size=size, duration=duration)
    composition.add_layer(source, end_time=time)
    composition.add_layer(target, offset=time)
    composition.add_layer(source, offset=target.duration, start_time=time)
    return composition
