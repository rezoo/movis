from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from movis.layer.composition import Composition
from movis.layer.media import Image
from movis.layer.protocol import BasicLayer


def _get_size(layer: BasicLayer, size: tuple[int, int] | None) -> tuple[int, int]:
    """Determines the size of the layer."""
    if size is not None:
        return size
    elif isinstance(layer, Composition):
        return layer.size
    img = layer(0.0)
    if img is None:
        raise ValueError("Cannot determine size of composition.")
    return img.shape[1], img.shape[0]


class _ConcatenateLayer:

    def __init__(self, layers: Sequence[BasicLayer]):
        self.layers = layers
        durations = np.array([layer.duration for layer in layers], dtype=np.float64)
        self.offsets = np.cumsum(np.concatenate([[0.], durations]))
        self._duration = self.offsets[-1]

    @property
    def duration(self) -> float:
        return self._duration

    def __len__(self) -> int:
        return len(self.layers)

    def get_state(self, time: float) -> tuple[int, float] | None:
        if time < 0.0 or self.duration <= time:
            return None
        ind = int(np.searchsorted(self.offsets, time, side="right") - 1)
        t = time - self.offsets[ind]
        return ind, t

    def __call__(self, time: float) -> np.ndarray | None:
        result = self.get_state(time)
        if result is None:
            return None
        ind, t = result
        return self.layers[ind](t)

    def get_key(self, time: float) -> Hashable:
        result = self.get_state(time)
        if result is None:
            return None
        ind, t = result
        return (ind, self.layers[ind](t))

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        c = Composition(size=(8, 8), duration=self.duration)
        time = 0.0
        for layer in self.layers:
            c.add_layer(layer, offset=time)
            time += layer.duration
        return c.get_audio(start_time, end_time)


def concatenate(layers: Sequence[BasicLayer]) -> _ConcatenateLayer:
    """Concatenate layers into a single composition.

    Args:
        layers:
            Layers to concatenate.

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
    return _ConcatenateLayer(layers)


class _RepeatLayer:

    def __init__(self, layer: BasicLayer, n_repeat: int, repeat_method: str):
        self.layer = layer
        self.n_repeat = n_repeat
        self.repeat_method = repeat_method

    @property
    def duration(self) -> float:
        return self.layer.duration * self.n_repeat

    def get_state(self, time: float) -> float | None:
        if time < 0.0 or self.duration <= time:
            return None

        completed_cycles = int(time / self.layer.duration)
        t = time % self.layer.duration

        if completed_cycles % 2 > 0 and self.repeat_method == "bounce":
            t = self.layer.duration - t

        return t

    def __call__(self, time: float) -> np.ndarray | None:
        t = self.get_state(time)
        if t is None:
            return None
        return self.layer(t)

    def get_key(self, time: float) -> Hashable:
        t = self.get_state(time)
        if t is None:
            return None
        return self.layer.get_key(t)

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        c = Composition(size=(8, 8), duration=self.duration)
        for i in range(self.n_repeat):
            c.add_layer(self.layer, offset=i * self.layer.duration)
        return c.get_audio(start_time, end_time)


def repeat(layer: BasicLayer, n_repeat: int, repeat_method: str = "loop") -> _RepeatLayer:
    """Repeat a layer multiple times.

    Args:
        layer:
            Layer to repeat.
        n_repeat:
            Number of times to repeat the layer.
        repeat_method:
            Method to repeat the layer, choose from bounce or loop.

    Returns:
        Composition with the layer repeated.

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Image("image.png", duration=1.0)
        >>> composition = mv.repeat(layer, 3)  # repeat 3 times
        >>> composition.duration
        3.0
    """

    return _RepeatLayer(layer, n_repeat, repeat_method)


class _TrimLayer:

    def __init__(self, layer: BasicLayer, start_times: Sequence[float], end_times: Sequence[float]):
        assert 0 < len(start_times)
        assert len(start_times) == len(end_times)
        self.layer = layer
        self.start_times = np.array(start_times, dtype=np.float64)
        self.end_times = np.array(end_times, dtype=np.float64)
        assert np.all(self.start_times < self.end_times)
        durations = self.end_times - self.start_times
        self.offsets = np.cumsum(np.concatenate([[0.], durations]))
        self._duration = self.offsets[-1]

    def __len__(self) -> int:
        return len(self.start_times)

    @property
    def duration(self) -> float:
        return self._duration

    def get_state(self, time: float) -> tuple[BasicLayer, float] | None:
        if time < 0.0 or self.duration <= time:
            return None
        ind = np.searchsorted(self.offsets, time, side="right") - 1
        delta = time - self.offsets[ind]
        t = self.start_times[ind] + delta
        return self.layer, t

    def __call__(self, time: float) -> np.ndarray | None:
        result = self.get_state(time)
        if result is None:
            return None
        layer, t = result
        return layer(t)

    def get_key(self, time: float) -> Hashable:
        result = self.get_state(time)
        if result is None:
            return None
        layer, t = result
        return layer.get_key(t)

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        c = Composition(size=(8, 8), duration=self.duration)
        offsets = self.offsets[:-1] - self.start_times
        for start, end, offset in zip(self.start_times, self.end_times, offsets):
            c.add_layer(self.layer, start_time=start, end_time=end, offset=offset)
        return c.get_audio(start_time, end_time)


def trim(
    layer: BasicLayer, start_times: Sequence[float], end_times: Sequence[float]
) -> _TrimLayer:
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
    return _TrimLayer(layer, start_times, end_times)


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
    w, h = _get_size(layers[0], size)
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


class _CropLayer:

    def __init__(self, layer: BasicLayer, rect: tuple[int, int, int, int]):
        assert len(rect) == 4
        self.layer = layer
        self.rect = rect

    @property
    def duration(self) -> float:
        return self.layer.duration

    def __call__(self, time: float) -> np.ndarray | None:
        img = self.layer(time)
        if img is None:
            return None
        x, y, w, h = self.rect
        return img[y:y+h, x:x+w]

    def get_key(self, time: float) -> Hashable:
        return self.layer.get_key(time)

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        if hasattr(self.layer, 'get_audio'):
            return self.layer.get_audio(start_time, end_time)
        else:
            return None


def crop(layer: BasicLayer, rect: tuple[int, int, int, int]) -> _CropLayer:
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
    return _CropLayer(layer, rect)


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
    size = _get_size(layers[0], size)
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

    .. raw:: html

        <pre>
        |------------------------|    |------------|----------|------------|
        |    Source layer        | -> |   Source   |  target  |   Source   |
        |------------------------|    |------------|----------|------------|
        </pre>

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
    """
    size = _get_size(source, size)
    duration = source.duration + target.duration
    composition = Composition(size=size, duration=duration)
    composition.add_layer(source, end_time=time)
    composition.add_layer(target, offset=time)
    composition.add_layer(source, offset=target.duration, start_time=time)
    return composition


def fade_in(
    layer: BasicLayer, duration: float = 0.0,
    size: tuple[int, int] | None = None,
    bg_color: tuple[int, int, int] | str | None = None,
) -> Composition:
    """Fade in a layer.

    Args:
        layer:
            Layer to fade in.
        duration:
            Duration of the fade-in effect.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.
        bg_color:
            Background color. If ``None``, the background is transparent.
    """
    return fade_in_out(layer, fade_in=duration, fade_out=0.0, size=size, bg_color=bg_color)


def fade_out(
    layer: BasicLayer, duration: float = 0.0,
    size: tuple[int, int] | None = None,
    bg_color: tuple[int, int, int] | str | None = None,
) -> Composition:
    """Fade out a layer.

    Args:
        layer:
            Layer to fade out.
        duration:
            Duration of the fade-out effect.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.
        bg_color:
            Background color. If ``None``, the background is transparent.
    """
    return fade_in_out(layer, fade_in=0.0, fade_out=duration, size=size, bg_color=bg_color)


def fade_in_out(
    layer: BasicLayer, fade_in: float = 0.0, fade_out: float = 0.0,
    size: tuple[int, int] | None = None,
    bg_color: tuple[int, int, int] | str | None = None,
) -> Composition:
    """Fade in and out a layer. If ``fade_in`` or ``fade_out`` is ``0.0``, the corresponding effect is not applied.

    Args:
        layer:
            Layer to fade in and out.
        fade_in:
            Duration of the fade-in effect.
        fade_out:
            Duration of the fade-out effect.
        size:
            Size of the composition. If ``None``, the size of the layer is estimated.
        bg_color:
            Background color. If ``None``, the background is transparent.

    Returns:
        Composition with the layer faded in and out.

    Examples:
        >>> import movis as mv
        >>> layer = mv.layer.Image.from_color((10, 10), "white", duration=3.0)
        >>> composition = mv.fade_in_out(layer, fade_in=1.0, fade_out=1.0)
        >>> composition(0.0)[0, 0, :]
        array([0, 0, 0, 0], dtype=uint8)
        >>> composition(1.0)[0, 0, :]
        array([255, 255, 255, 255], dtype=uint8)
        >>> composition(2.0)[0, 0, :]
        array([255, 255, 255, 255], dtype=uint8)
        >>> composition(3.0 - 1e-5)[0, 0, :]
        array([0, 0, 0, 0], dtype=uint8)
    """
    assert 0.0 <= fade_in, "fade_in must be non-negative."
    assert 0.0 <= fade_out, "fade_out must be non-negative."
    assert fade_in + fade_out <= layer.duration, \
        "fade_in + fade_out must be less than or equal to the layer duration."
    size = _get_size(layer, size)
    composition = Composition(size=size, duration=layer.duration)
    if bg_color is not None:
        composition.add_layer(
            Image.from_color(size, bg_color, duration=layer.duration), name='bg')
    item = composition.add_layer(layer, name="main")
    if 0.0 < fade_in:
        item.opacity.enable_motion().extend(
            keyframes=[0.0, fade_in], values=[0.0, 1.0])
        item.audio_level.enable_motion().extend(
            keyframes=[0.0, fade_in], values=[0.0, -40.0])
    if 0.0 < fade_out:
        item.opacity.enable_motion().extend(
            keyframes=[layer.duration - fade_out, layer.duration], values=[1.0, 0.0])
        item.audio_level.enable_motion().extend(
            keyframes=[layer.duration - fade_out, layer.duration], values=[-40.0, 0.0])
    return composition
