from enum import Enum
from pathlib import Path
from typing import Hashable, Optional, Union

import imageio
import numpy as np
from cachetools import LRUCache
from tqdm import tqdm

from zunda.attribute import Attribute
from zunda.imgproc import alpha_composite, resize
from zunda.layer.core import Layer
from zunda.effect import Effect
from zunda.transform import Transform


class CacheType(Enum):
    COMPOSITION = 0
    LAYER = 1


class LayerItem:

    def __init__(
            self, name: str, layer: Layer, transform: Optional[Transform] = None,
            offset: float = 0.0, start_time: float = 0.0, end_time: float = 0.0):
        self.name: str = name
        self.layer: Layer = layer
        self.transform: Transform = transform if transform is not None else Transform()
        self.offset: float = offset
        self.start_time: float = start_time
        self.end_time: float = end_time if end_time == 0.0 else self.layer.duration
        self._effects: list[Effect] = []

    @property
    def attributes(self) -> dict[str, Union[tuple[dict[str, Attribute], ...], dict[str, Attribute]]]:
        return {
            'transform': self.transform.attributes,
            'effects': tuple(effect.attributes for effect in self._effects),
        }

    def get_key(self, layer_time: float) -> tuple[Hashable, ...]:
        p = self.transform.get_current_value(layer_time)
        return (p, self.layer.get_key(layer_time))

    def __call__(self, layer_time: float) -> Optional[np.ndarray]:
        frame = self.layer(layer_time)
        if frame is None:
            return None
        for effect in self._effects:
            frame = effect(layer_time, frame)
        return frame


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

    @property
    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def __getitem__(self, key: str) -> LayerItem:
        return self._name_to_layer[key]

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        layer_keys: list[Hashable] = [CacheType.COMPOSITION]
        for layer_item in self.layers:
            layer_time = time - layer_item.offset
            if layer_time < layer_item.start_time or layer_item.end_time <= layer_time:
                layer_keys.append(f"__{layer_item.name}__")
            else:
                layer_keys.append(layer_item.get_key(layer_time))
        return tuple(layer_keys)

    def add_layer(
        self,
        layer: Layer,
        name: Optional[str] = None,
        transform: Optional[Transform] = None,
        offset: float = 0.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> LayerItem:
        if name is None:
            name = f"layer_{len(self.layers)}"
        if name in self.layers:
            raise KeyError(f"Layer with name {name} already exists")
        end_time = end_time if end_time is not None else layer.duration
        transform = transform if transform is not None else Transform()
        layer_item = LayerItem(
            name,
            layer,
            transform,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
        )
        self.layers.append(layer_item)
        self._name_to_layer[name] = layer_item
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
        component = self._get_or_resize(layer_item, layer_time, component, p.scale)
        x = p.position[0] + (p.anchor_point[0] - w / 2) * p.scale[0]
        y = p.position[1] + (p.anchor_point[1] - h / 2) * p.scale[1]
        base_img = alpha_composite(
            base_img, component, position=(round(x), round(y)), opacity=p.opacity)
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
        fps: float = 30.0,
    ) -> None:
        if end_time is None:
            end_time = self.duration
        times = np.arange(start_time, end_time, 1.0 / fps)
        writer = imageio.get_writer(
            dst_file, fps=fps, codec=codec, macro_block_size=None
        )
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self(t))
            writer.append_data(frame)
        writer.close()
        self.cache.clear()
