from typing import Hashable, Optional, Union

import numpy as np

from zunda.effect import Effect
from zunda.imgproc import BlendingMode
from zunda.layer.layer import Layer
from zunda.transform import Transform


class Component:

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
    def composition_start_time(self) -> float:
        return self.offset + self.start_time

    @property
    def composition_end_time(self) -> float:
        return self.offset + self.end_time

    def add_effect(self, effect: Effect) -> Effect:
        self._effects.append(effect)
        return effect

    def get_key(self, layer_time: float) -> tuple[Hashable, Hashable, Hashable]:
        if not self.visible:
            return (None, None, None)
        transform_key = self.transform.get_current_value(layer_time)
        layer_key = self.layer.get_key(layer_time) if hasattr(self.layer, 'get_key') else layer_time

        def get_effect_key(e: Effect) -> Optional[Hashable]:
            return e.get_key(layer_time) if hasattr(e, 'get_key') else layer_time

        effects_key = None if len(self._effects) == 0 else tuple([get_effect_key(e) for e in self._effects])
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
