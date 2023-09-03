from __future__ import annotations
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Hashable, Iterator, Sequence
from weakref import WeakValueDictionary

import imageio
import numpy as np
from diskcache import Cache
from tqdm import tqdm

from ..enum import CacheType
from ..transform import Transform
from .layer import Layer
from .layer_item import LayerItem


class Composition:
    """A base layer that integrates multiple layers.

    Users create a composition by specifying both time and resolution. Next, multiple layers can be added to
    the target composition through `Composition.add_layer()`. During this process, additional information such as
    the layer's name, start time, position, opacity, and drawing mode can be specified.
    Finally, the composition integrates the layers in the order they were added to create a single video.

    Another composition can also be added as a layer within a composition.
    By nesting compositions in this way, more complex motions can be created.

    Args:
        size: A tuple representing the size of the composition in the form of `(width, height)`.
        duration: The duration along the time axis for the composition.
    """

    def __init__(
        self, size: tuple[int, int] = (1920, 1080), duration: float = 1.0
    ) -> None:
        self._layers: list[LayerItem] = []
        self._name_to_layer: WeakValueDictionary[str, LayerItem] = WeakValueDictionary()
        self._duration = duration
        self._cache: Cache = Cache(size_limit=1024 * 1024 * 1024)
        self._preview_level: int = 1
        self._size = size

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0
        self._size = (int(size[0]), int(size[1]))

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        assert duration > 0
        self._duration = float(duration)

    @property
    def preview_level(self) -> int:
        return self._preview_level

    @preview_level.setter
    def preview_level(self, level: int) -> None:
        assert level > 0
        self._preview_level = int(level)

    @contextmanager
    def preview(self, level: int = 2) -> Iterator[None]:
        assert level > 0
        original_level = self._preview_level
        self._preview_level = level
        try:
            yield
        finally:
            self._preview_level = original_level

    @contextmanager
    def final(self) -> Iterator[None]:
        original_level = self._preview_level
        self._preview_level = 1
        try:
            yield
        finally:
            self._preview_level = original_level

    @property
    def layers(self) -> Sequence[LayerItem]:
        return self._layers

    def keys(self) -> list[str]:
        """Returns a list of layer names.

        Note that the keys are sorted in the order in which they will be rendered.

        Returns:
            List[str]: A list of layer names sorted in the rendering order.
        """
        return [layer.name for layer in self._layers]

    def values(self) -> list[LayerItem]:
        """Returns a list of LayerItem objects.

        Note that the elements of the list are not the layers themselves,
        but `LayerItem` containing information of the layers.

        Returns:
            List[LayerItem]: A list of `LayerItem` objects.
        """
        return self._layers

    def items(self) -> list[tuple[str, LayerItem]]:
        """Returns a list of tuples, each consisting of a layer name and its corresponding item.

        Returns:
            List[Tuple[str, LayerItem]]: A list of tuples, where each tuple contains a layer name and its layer item.
        """
        return [(layer.name, layer) for layer in self._layers]

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, key: str) -> LayerItem:
        return self._name_to_layer[key]

    def __setitem__(self, key: str, value: LayerItem | Layer) -> None:
        if isinstance(value, LayerItem):
            self._layers.append(value)
            self._name_to_layer[key] = value
        elif callable(value):
            self.add_layer(value, name=key)
        else:
            raise ValueError("value must be LayerItem or Layer (i.e., callable)")

    def __delitem__(self, key: str) -> None:
        self.pop_layer(key)

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        layer_keys: list[Hashable] = [CacheType.COMPOSITION]
        for layer_item in self._layers:
            layer_time = time - layer_item.offset
            if layer_time < layer_item.start_time or layer_item.end_time <= layer_time:
                layer_keys.append(None)
            else:
                layer_keys.append(layer_item.get_key(layer_time))
        return tuple(layer_keys)

    def __repr__(self) -> str:
        return f"Composition(size={self.size}, duration={self.duration}, layers={self._layers!r})"

    def add_layer(
        self,
        layer: Layer,
        name: str | None = None,
        transform: Transform | None = None,
        offset: float = 0.0,
        start_time: float = 0.0,
        end_time: float | None = None,
        visible: bool = True,
    ) -> LayerItem:
        if name is None:
            name = f"layer_{len(self._layers)}"
        if name in self._name_to_layer:
            raise KeyError(f"Layer with name {name} already exists")
        end_time = end_time if end_time is not None else getattr(layer, "duration", 1e6)
        transform = transform if transform is not None \
            else Transform.from_positions(self.size)
        layer_item = LayerItem(
            layer,
            name,
            transform,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
            visible=visible,
        )
        self._layers.append(layer_item)
        self._name_to_layer[name] = layer_item
        return layer_item

    def pop_layer(self, name: str) -> LayerItem:
        if name not in self._name_to_layer:
            raise KeyError(f"Layer with name {name} does not exist")
        index = next(i for i in range(len(self._layers)) if self._layers[i].name == name)
        layer_item = self._layers.pop(index)
        return layer_item

    def __call__(self, time: float) -> np.ndarray | None:
        L = self._preview_level
        current_shape = self.size[1] // L, self.size[0] // L

        key = self.get_key(time)
        if key in self._cache:
            cached_frame: np.ndarray = self._cache[key]
            if cached_frame.shape[:2] == current_shape:
                return cached_frame
            else:
                del self._cache[key]

        frame = np.zeros(current_shape + (4,), dtype=np.uint8)
        for layer_item in self._layers:
            frame = layer_item._composite(
                frame, time, preview_level=self._preview_level)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 4
            assert frame.dtype == np.uint8
        self._cache[key] = frame
        return frame

    def write_video(
        self,
        dst_file: str | Path,
        start_time: float = 0.0,
        end_time: float | None = None,
        codec: str = "libx264",
        pixelformat: str = "yuv420p",
        fps: float = 30.0,
        audio_path: str | Path | None = None,
    ) -> None:
        if end_time is None:
            end_time = self.duration
        times = np.arange(start_time, end_time, 1.0 / fps)
        if audio_path is not None:
            audio_path = str(audio_path)
        writer = imageio.get_writer(
            dst_file, fps=fps, codec=codec, pixelformat=pixelformat,
            macro_block_size=None, audio_path=audio_path,
            ffmpeg_log_level="error",
        )
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self(t))
            writer.append_data(frame)
        writer.close()
        self._cache.clear()

    def render_and_play(
        self,
        start_time: float = 0.0,
        end_time: float | None = None,
        fps: float = 30.0,
        preview_level: int = 2
    ) -> None:
        from IPython.display import display
        from ipywidgets import Video

        if end_time is None:
            end_time = self.duration

        times = np.arange(start_time, end_time, 1.0 / fps)
        with tempfile.NamedTemporaryFile(suffix='.mp4') as fp:
            with self.preview(level=preview_level):
                filename: str = fp.name
                writer = imageio.get_writer(
                    filename, fps=fps, codec="libx264",
                    ffmpeg_params=["-preset", "veryfast"],
                    pixelformat="yuv444p", macro_block_size=None,
                    ffmpeg_log_level="error")
                for t in tqdm(times, total=len(times)):
                    frame = np.asarray(self(t))
                    writer.append_data(frame)
                writer.close()
                self._cache.clear()

                display(Video.from_file(filename, autoplay=True, loop=True))
