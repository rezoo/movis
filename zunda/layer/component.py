from typing import Hashable, Optional, Union

import numpy as np
from cachetools import LRUCache

from zunda.effect import Effect
from zunda.enum import CacheType
from zunda.imgproc import BlendingMode, alpha_composite, resize
from zunda.layer.layer import Layer
from zunda.transform import Transform


class Component:

    def __init__(
            self, name: str, layer: Layer, transform: Optional[Transform] = None,
            offset: float = 0.0, start_time: float = 0.0, end_time: float = 0.0,
            visible: bool = True, blending_mode: Union[BlendingMode, str] = BlendingMode.NORMAL,
            alpha_matte: Optional["Component"] = None):
        self.name: str = name
        self.layer: Layer = layer
        self.transform: Transform = transform if transform is not None else Transform()
        self.offset: float = offset
        self.start_time: float = start_time
        self.end_time: float = end_time if end_time == 0.0 else self.layer.duration
        self.visible: bool = visible
        mode = BlendingMode.from_string(blending_mode) if isinstance(blending_mode, str) else blending_mode
        self.blending_mode: BlendingMode = mode
        self._alpha_matte: Optional[Component] = alpha_matte
        self._effects: list[Effect] = []

    @property
    def composition_start_time(self) -> float:
        return self.offset + self.start_time

    @property
    def composition_end_time(self) -> float:
        return self.offset + self.end_time

    def enable_alpha_matte(self, alpha_matte: "Component") -> "Component":
        self._alpha_matte = alpha_matte
        return alpha_matte

    def disable_alpha_matte(self) -> Optional["Component"]:
        if self._alpha_matte is None:
            return None
        prev_alpha_matte = self._alpha_matte
        self._alpha_matte = None
        return prev_alpha_matte

    def add_effect(self, effect: Effect) -> Effect:
        self._effects.append(effect)
        return effect

    def get_key(self, layer_time: float) -> tuple[Hashable, Hashable, Hashable, Hashable]:
        if not self.visible:
            return (None, None, None, None)
        transform_key = self.transform.get_current_value(layer_time)
        layer_key = self.layer.get_key(layer_time) if hasattr(self.layer, 'get_key') else layer_time
        alpha_matte_key = self._alpha_matte.get_key(layer_time) \
            if self._alpha_matte is not None and hasattr(self._alpha_matte, 'get_key') else None

        def get_effect_key(e: Effect) -> Optional[Hashable]:
            return e.get_key(layer_time) if hasattr(e, 'get_key') else layer_time

        effects_key = None if len(self._effects) == 0 else tuple([get_effect_key(e) for e in self._effects])
        return (transform_key, layer_key, alpha_matte_key, effects_key)

    def _get_or_resize(
        self, layer_time: float, image: np.ndarray, scale: tuple[float, float],
        cache: Optional[LRUCache] = None,
    ) -> np.ndarray:
        if scale == 1.0:
            return image
        layer_state = self.get_key(layer_time)
        key = (CacheType.LAYER, self.name, layer_state, scale)
        if cache is not None and key in cache:
            return cache[key]
        resized_image = resize(image, scale)
        if cache is not None:
            cache[key] = resized_image
        return resized_image

    def _composite(
        self, bg_image: np.ndarray, time: float,
        parent: tuple[float, float] = (0.0, 0.0),
        cache: Optional[LRUCache] = None,
    ) -> np.ndarray:
        layer_time = time - self.offset
        if layer_time < self.start_time or self.end_time <= layer_time:
            return bg_image
        fg_image = self(layer_time)
        if fg_image is None:
            return bg_image
        h, w = fg_image.shape[:2]

        p = self.transform.get_current_value(layer_time)
        if round(fg_image.shape[1] * p.scale[0]) == 0 or round(fg_image.shape[0] * p.scale[1]) == 0:
            return bg_image
        fg_image = self._get_or_resize(layer_time, fg_image, p.scale, cache)
        x = p.position[0] + (p.anchor_point[0] - w / 2) * p.scale[0] - parent[0]
        y = p.position[1] + (p.anchor_point[1] - h / 2) * p.scale[1] - parent[1]
        bg_image = alpha_composite(
            bg_image, fg_image, position=(round(x), round(y)),
            opacity=p.opacity, blending_mode=self.blending_mode)
        return bg_image

    def __call__(self, layer_time: float) -> Optional[np.ndarray]:
        if not self.visible:
            return None
        frame = self.layer(layer_time)
        if frame is None:
            return None
        for effect in self._effects:
            frame = effect(layer_time, frame)

        if self._alpha_matte is not None:
            p = self.transform.get_current_value(layer_time)
            if round(frame.shape[1] * p.scale[0]) == 0 or round(frame.shape[0] * p.scale[1]) == 0:
                return frame
            x = p.position[0] + (p.anchor_point[0] - frame.shape[1] / 2) * p.scale[0]
            y = p.position[1] + (p.anchor_point[1] - frame.shape[0] / 2) * p.scale[1]
            time = layer_time + self.offset
            frame = self._alpha_matte._composite(frame, time, parent=(x, y))
        return frame

    def __repr__(self) -> str:
        return f"LayerItem(name={self.name!r}, layer={self.layer!r}, transform={self.transform!r}, " \
            f"offset={self.offset}, blending_mode={self.blending_mode})"
