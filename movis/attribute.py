from __future__ import annotations

from typing import Callable, Hashable

import numpy as np

from movis.enum import AttributeType
from movis.motion import Motion, transform_to_hashable, transform_to_numpy


class Attribute:
    def __init__(
        self,
        init_value: float | tuple[float, ...] | np.ndarray,
        value_type: AttributeType,
        range: tuple[float, float] | None = None,
        motion: Motion | None = None,
        function: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> None:
        np_value = transform_to_numpy(init_value, value_type)
        clipped_value = np.clip(np_value, range[0], range[1]) if range is not None else np_value
        self.init_value: np.ndarray = clipped_value
        self.value_type = value_type
        self.range = range
        self._motion = motion
        self._function = function

    def __call__(self, layer_time: float) -> np.ndarray:
        if self._motion is None:
            return transform_to_numpy(self.init_value, self.value_type)
        else:
            value = self.init_value
            if self._motion is not None:
                value = self._motion(value, layer_time)
            if self._function is not None:
                value = self._function(value, layer_time)
            np_value = transform_to_numpy(value, self.value_type)
            if self.range is not None:
                return np.clip(np_value, self.range[0], self.range[1])
            else:
                return np_value

    def enable_motion(self) -> Motion:
        if self._motion is None:
            motion = Motion(init_value=self.init_value, value_type=self.value_type)
            self._motion = motion
        return self._motion

    def disable_motion(self) -> None:
        self._motion = None

    @property
    def motion(self) -> Motion | None:
        return self._motion

    def enable_function(
        self, function: Callable[[np.ndarray, float], np.ndarray]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        if not callable(function):
            raise ValueError(f"Invalid function: {function}")
        self._function = function
        return function

    def disable_function(self) -> None:
        self._function = None

    @property
    def function(self) -> Callable[[np.ndarray, float], np.ndarray] | None:
        return self._function

    def __repr__(self) -> str:
        if self._motion is None:
            return f"{self.init_value}"
        else:
            return f"Attribute(value_type={self.value_type})"


class AttributesMixin:
    @property
    def attributes(self) -> dict[str, Attribute]:
        return {key: attr for key, attr in vars(self).items() if isinstance(attr, Attribute)}

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        return tuple([transform_to_hashable(attr(time)) for attr in self.attributes.values()])
