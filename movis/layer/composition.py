from pathlib import Path
from typing import Hashable, Optional, Sequence, Union

import imageio
import numpy as np
from cachetools import LRUCache
from tqdm import tqdm

from ..enum import CacheType, Direction
from ..imgproc import BlendingMode
from .component import Component
from .layer import Layer
from ..transform import Transform


class Composition:
    def __init__(
        self, size: tuple[int, int] = (1920, 1080), duration: float = 1.0
    ) -> None:
        self.layers: list[Component] = []
        self._name_to_layer: dict[str, Component] = {}
        self.size = size
        self._duration = duration
        self.cache: LRUCache = LRUCache(maxsize=128)

    @property
    def duration(self) -> float:
        return self._duration

    def keys(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def values(self) -> list[Component]:
        return self.layers

    def items(self) -> list[tuple[str, Component]]:
        return [(layer.name, layer) for layer in self.layers]

    def __getitem__(self, key: str) -> Component:
        return self._name_to_layer[key]

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        layer_keys: list[Hashable] = [CacheType.COMPOSITION]
        for component in self.layers:
            layer_time = time - component.offset
            if layer_time < component.start_time or component.end_time <= layer_time:
                layer_keys.append(None)
            else:
                layer_keys.append(component.get_key(layer_time))
        return tuple(layer_keys)

    def __repr__(self) -> str:
        return f"Composition(size={self.size}, duration={self.duration}, layers={self.layers!r})"

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
    ) -> Component:
        if name is None:
            name = f"layer_{len(self.layers)}"
        if name in self.layers:
            raise KeyError(f"Layer with name {name} already exists")
        end_time = end_time if end_time is not None else layer.duration
        transform = transform if transform is not None \
            else Transform(position=(self.size[0] / 2, self.size[1] / 2))
        component = Component(
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
        self.layers.append(component)
        self._name_to_layer[name] = component
        return component

    def pop_layer(self, name: str) -> Component:
        if name not in self._name_to_layer:
            raise KeyError(f"Layer with name {name} does not exist")
        index = next(i for i in range(len(self.layers)) if self.layers[i].name == name)
        self.layers.pop(index)
        component = self._name_to_layer.pop(name)
        return component

    def enable_alpha_matte(self, source_name: str, target_name: str) -> Component:
        assert source_name in self._name_to_layer and target_name in self._name_to_layer, \
            "source and target must be in self.layers"
        target = self.pop_layer(target_name)
        source = self[source_name]
        source.enable_alpha_matte(target)
        return source

    def disable_alpha_matte(self, name: str) -> Optional[Component]:
        source = self[name]
        target = source.disable_alpha_matte()
        return target

    def __call__(self, time: float) -> Optional[np.ndarray]:
        key = self.get_key(time)
        if key in self.cache:
            return self.cache[key]

        frame = np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        for component in self.layers:
            frame = component._composite(frame, time)
        self.cache[key] = frame
        return frame

    def write_video(
        self,
        dst_file: Union[str, Path],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        codec: str = "libx264",
        pixelformat: str = "yuv420p",
        fps: float = 30.0,
    ) -> None:
        if end_time is None:
            end_time = self.duration
        times = np.arange(start_time, end_time, 1.0 / fps)
        writer = imageio.get_writer(
            dst_file, fps=fps, codec=codec, pixelformat=pixelformat, macro_block_size=None
        )
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self(t))
            writer.append_data(frame)
        writer.close()
        self.cache.clear()


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
