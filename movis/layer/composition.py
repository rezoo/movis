import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Hashable, Iterator, Optional, Sequence, Union
from weakref import WeakValueDictionary

import imageio
import numpy as np
from diskcache import Cache
from tqdm import tqdm

from ..enum import CacheType, Direction
from ..imgproc import BlendingMode
from ..transform import Transform
from .layer_item import LayerItem
from .layer import Layer


class Composition:
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
        self._size = size

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        assert duration > 0
        self._duration = duration

    @property
    def preview_level(self) -> int:
        return self._preview_level

    @preview_level.setter
    def preview_level(self, level: int) -> None:
        assert level > 0
        self._preview_level = level

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
        return [layer.name for layer in self._layers]

    def values(self) -> list[LayerItem]:
        return self._layers

    def items(self) -> list[tuple[str, LayerItem]]:
        return [(layer.name, layer) for layer in self._layers]

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, key: str) -> LayerItem:
        return self._name_to_layer[key]

    def __setitem__(self, key: str, value: Union[LayerItem, Layer]) -> None:
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
        name: Optional[str] = None,
        transform: Optional[Transform] = None,
        offset: float = 0.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        visible: bool = True,
        blending_mode: Union[BlendingMode, str] = BlendingMode.NORMAL,
        origin_point: Direction = Direction.CENTER,
    ) -> LayerItem:
        if name is None:
            name = f"layer_{len(self._layers)}"
        if name in self._layers:
            raise KeyError(f"Layer with name {name} already exists")
        end_time = end_time if end_time is not None else layer.duration
        transform = transform if transform is not None \
            else Transform(position=(self.size[0] / 2, self.size[1] / 2))
        layer_item = LayerItem(
            name,
            layer,
            transform,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
            visible=visible,
            blending_mode=blending_mode,
            origin_point=origin_point,
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

    def enable_alpha_matte(self, source_name: str, target_name: str) -> LayerItem:
        assert source_name in self._name_to_layer and target_name in self._name_to_layer, \
            "source and target must be in self.layers"
        target = self.pop_layer(target_name)
        source = self[source_name]
        source.enable_alpha_matte(target)
        return source

    def disable_alpha_matte(self, name: str) -> Optional[LayerItem]:
        source = self[name]
        target = source.disable_alpha_matte()
        return target

    def __call__(self, time: float) -> Optional[np.ndarray]:
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
        self._cache[key] = frame
        return frame

    def write_video(
        self,
        dst_file: Union[str, Path],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        codec: str = "libx264",
        pixelformat: str = "yuv420p",
        fps: float = 30.0,
        audio_path: Optional[Union[str, Path]] = None,
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
        end_time: Optional[float] = None,
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
