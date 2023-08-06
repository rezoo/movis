from enum import Enum
from pathlib import Path
from typing import Hashable, Optional, Sequence, Union

import imageio
import numpy as np
from cachetools import LRUCache
from tqdm import tqdm

from zunda.attribute import Attribute
from zunda.effect import Effect
from zunda.imgproc import BlendingMode, alpha_composite, resize
from zunda.layer.core import Layer
from zunda.transform import Transform


class CacheType(Enum):
    COMPOSITION = 0
    LAYER = 1


class LayerItem:

    def __init__(
            self, name: str, layer: Layer, transform: Optional[Transform] = None,
            offset: float = 0.0, start_time: float = 0.0, end_time: float = 0.0,
            visible: bool = True, blending_mode: Union[BlendingMode, str] = BlendingMode.NORMAL):
        self.name: str = name
        self.layer: Layer = layer
        self.transform: Transform = transform if transform is not None else Transform()
        self.offset: float = offset
        self.start_time: float = start_time
        self.end_time: float = end_time if end_time == 0.0 else self.layer.duration
        self.visible: bool = visible
        mode = BlendingMode.from_string(blending_mode) if isinstance(blending_mode, str) else blending_mode
        self.blending_mode: BlendingMode = mode
        self._effects: list[Effect] = []

    @property
    def attributes(self) -> dict[str, Union[tuple[dict[str, Attribute], ...], dict[str, Attribute]]]:
        return {
            'transform': self.transform.attributes,
            'effects': tuple(effect.attributes for effect in self._effects),
        }

    @property
    def composition_start_time(self) -> float:
        return self.offset + self.start_time

    @property
    def composition_end_time(self) -> float:
        return self.offset + self.end_time

    def get_key(self, layer_time: float) -> tuple[Hashable, ...]:
        if not self.visible:
            return (None, None, None)
        transform_key = self.transform.get_current_value(layer_time)
        layer_key = self.layer.get_key(layer_time)
        effects_key = None if len(self._effects) == 0 else tuple([e.get_key(layer_time) for e in self._effects])
        return (transform_key, layer_key, effects_key)

    def __call__(self, layer_time: float) -> Optional[np.ndarray]:
        if not self.visible:
            return None
        frame = self.layer(layer_time)
        if frame is None:
            return None
        for effect in self._effects:
            frame = effect(layer_time, frame)
        return frame

    def __repr__(self) -> str:
        return f"LayerItem(name={self.name!r}, layer={self.layer!r}, transform={self.transform!r}, " \
            f"offset={self.offset}, blending_mode={self.blending_mode})"


class Composition:
    def __init__(
        self, size: tuple[int, int] = (1920, 1080), duration: float = 1.0
    ) -> None:
        self.layers: list[LayerItem] = []
        self._name_to_layer: dict[str, LayerItem] = {}
        self.size = size
        self._duration = duration
        self.cache: LRUCache = LRUCache(maxsize=128)

    @property
    def duration(self) -> float:
        return self._duration

    def keys(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def values(self) -> list[LayerItem]:
        return self.layers

    def items(self) -> list[tuple[str, LayerItem]]:
        return [(layer.name, layer) for layer in self.layers]

    def __getitem__(self, key: str) -> LayerItem:
        return self._name_to_layer[key]

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        layer_keys: list[Hashable] = [CacheType.COMPOSITION]
        for layer_item in self.layers:
            layer_time = time - layer_item.offset
            if layer_time < layer_item.start_time or layer_item.end_time <= layer_time:
                layer_keys.append(None)
            else:
                layer_keys.append(layer_item.get_key(layer_time))
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
    ) -> LayerItem:
        if name is None:
            name = f"layer_{len(self.layers)}"
        if name in self.layers:
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
        )
        self.layers.append(layer_item)
        self._name_to_layer[name] = layer_item
        return layer_item

    def pop_layer(self, name: str) -> LayerItem:
        if name not in self._name_to_layer:
            raise KeyError(f"Layer with name {name} does not exist")
        index = next(i for i in range(len(self.layers)) if self.layers[i].name == name)
        self.layers.pop(index)
        layer_item = self._name_to_layer.pop(name)
        return layer_item

    def _get_or_resize(
        self, layer_item: LayerItem, layer_time: float,
        component: np.ndarray, scale: tuple[float, float]
    ) -> np.ndarray:
        layer_state = layer_item.layer.get_key(layer_time)
        key = (CacheType.LAYER, layer_item.name, layer_state, scale)
        if key in self.cache:
            return self.cache[key]
        img = resize(component, scale)
        self.cache[key] = img
        return img

    def composite(
        self, base_img: np.ndarray, layer_item: LayerItem, time: float
    ) -> np.ndarray:
        layer_time = time - layer_item.offset
        if layer_time < layer_item.start_time or layer_item.end_time <= layer_time:
            return base_img
        component = layer_item(layer_time)
        if component is None:
            return base_img
        h, w = component.shape[:2]

        p = layer_item.transform.get_current_value(layer_time)
        if round(component.shape[1] * p.scale[0]) == 0 or round(component.shape[0] * p.scale[1]) == 0:
            return base_img
        component = self._get_or_resize(layer_item, layer_time, component, p.scale)
        x = p.position[0] + (p.anchor_point[0] - w / 2) * p.scale[0]
        y = p.position[1] + (p.anchor_point[1] - h / 2) * p.scale[1]
        base_img = alpha_composite(
            base_img, component, position=(round(x), round(y)),
            opacity=p.opacity, blending_mode=layer_item.blending_mode)
        return base_img

    def __call__(self, time: float) -> Optional[np.ndarray]:
        key = self.get_key(time)
        if key in self.cache:
            return self.cache[key]

        frame = np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        for layer_item in self.layers:
            frame = self.composite(frame, layer_item, time)
        self.cache[key] = frame
        return frame

    def make_video(
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
