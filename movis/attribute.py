from __future__ import annotations
from typing import Callable, Hashable, Sequence

import numpy as np

from movis.enum import AttributeType
from movis.motion import Motion, transform_to_hashable, transform_to_numpy


class Attribute:
    """Attribute Class for Animating Properties

    This class is used for animating layer properties.
    The dimensionality of the values that each Attribute can handle varies
    depending on the type of property.
    This is specified by setting an appropriate value for `value_type` using `AttributeType` Enum.
    For example, the values will be one-dimensional if `value_type` is set to `AttributeType.SCALAR`.
    If set to `AttributeType.COLOR`, the values will be three-dimensional.
    Regardless of `value_type`, the returned type will always be `np.ndarray`.
    Note that even if it's scalar, the returned value will be an array like `np.array([value])`.

    Args:
        init_value:
            The initial value to use. It is used when no animation is set,
            or when an animation with zero keyframes is set.
        value_type:
            Specifies the type of value that the Attribute will handle.
            For more details, see the docs of `AttributeType`.
        range:
            Defines the upper and lower limits of the possible values.
            If set, the returned array's values are guaranteed to be within this range;
            values that exceed this range are simply clipped.
        motion:
            The instance of the motion to use when adding keyframe animations.
            It can be specified in the constructor or activated later using `enable_motion()`.
        function:
            A user-defined function for adding animations, separate from keyframe animations.
            It can be specified in the constructor or added later using `add_function()`.
    """
    def __init__(
        self,
        init_value: float | tuple[float, ...] | np.ndarray,
        value_type: AttributeType,
        range: tuple[float, float] | None = None,
        motion: Motion | None = None,
        functions: Sequence[Callable[[np.ndarray, float], np.ndarray]] | None = None,
    ) -> None:
        np_value = transform_to_numpy(init_value, value_type)
        clipped_value = np.clip(np_value, range[0], range[1]) if range is not None else np_value
        self.init_value: np.ndarray = clipped_value
        self.value_type = value_type
        self.range = range
        self._motion = motion
        self._functions = [] if functions is None else list(functions)

    def __call__(self, layer_time: float) -> np.ndarray:
        if self._motion is None:
            return transform_to_numpy(self.init_value, self.value_type)
        else:
            value = self.init_value
            if self._motion is not None:
                value = transform_to_numpy(
                    self._motion(value, layer_time), self.value_type)
            if 0 < len(self._functions):
                for func in self._functions:
                    value = transform_to_numpy(
                        func(value, layer_time), self.value_type)
            if self.range is not None:
                return np.clip(value, self.range[0], self.range[1])
            else:
                return value

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

    @property
    def functions(self) -> list[Callable[[np.ndarray, float], np.ndarray]]:
        return self._functions

    def add_function(
        self, function: Callable[[np.ndarray, float], np.ndarray]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        if not callable(function):
            raise ValueError(f"Invalid function: {function}")
        self._functions.append(function)
        return function

    def pop_function(self, index: int) -> Callable[[np.ndarray, float], np.ndarray]:
        return self._functions.pop(index)

    def clear_functions(self) -> None:
        self._functions.clear()

    def __getitem__(self, index: int) -> Callable[[np.ndarray, float], np.ndarray]:
        return self._functions[index]

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
