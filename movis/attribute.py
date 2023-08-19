from __future__ import annotations

from typing import Callable, Hashable, Optional, Sequence, Union

import numpy as np

from movis.enum import AttributeType
from movis.motion import Motion


def transform_to_numpy(value: Union[int, float, Sequence[float], np.ndarray], value_type: AttributeType) -> np.ndarray:
    if isinstance(value, (int, float)):
        if value_type in (AttributeType.SCALAR, AttributeType.ANGLE):
            return np.array([value], dtype=np.float64)
        elif value_type == AttributeType.VECTOR2D:
            return np.array([value, value], dtype=np.float64)
        elif value_type in (AttributeType.VECTOR3D, AttributeType.COLOR):
            return np.array([value, value, value], dtype=np.float64)
    elif isinstance(value, (Sequence, np.ndarray)):
        if len(value) == 2 and value_type == AttributeType.VECTOR2D or \
                len(value) == 3 and value_type in (AttributeType.VECTOR3D, AttributeType.COLOR) or \
                len(value) == 1 and value_type in (AttributeType.SCALAR, AttributeType.ANGLE):
            return np.array(value, dtype=np.float64)
        else:
            raise ValueError(f"Invalid value type: {value_type}")
    raise ValueError(f"Invalid value type: {value_type}")


def transform_to_1dscalar(x: Union[float, Sequence[float], np.ndarray]) -> float:
    if isinstance(x, float):
        return x
    elif isinstance(x, np.ndarray):
        return float(x)
    else:
        return float(x[0])


def transform_to_2dvector(
    x: Union[float, Sequence[float], np.ndarray]
) -> tuple[float, float]:
    if isinstance(x, float):
        return (x, x)
    elif isinstance(x, np.ndarray) and x.shape == ():
        return (float(x), float(x))
    elif len(x) == 1:
        return (float(x[0]), float(x[0]))
    elif len(x) == 2:
        return (float(x[0]), float(x[1]))
    else:
        raise ValueError(f"Invalid value: {x}")


def transform_to_3dvector(
    x: Union[float, Sequence[float], np.ndarray]
) -> tuple[float, float, float]:
    if isinstance(x, float):
        return (x, x, x)
    elif isinstance(x, np.ndarray) and x.shape == ():
        return (float(x), float(x), float(x))
    elif len(x) == 1:
        y = float(x[0])
        return (y, y, y)
    elif len(x) == 3:
        return (float(x[0]), float(x[1]), float(x[2]))
    else:
        raise ValueError(f"Invalid value: {x}")


def transform_to_hashable(
    x: Union[float, Sequence[float], np.ndarray]
) -> Union[float, tuple[float, ...]]:
    if isinstance(x, (int, float)):
        return float(x)
    elif len(x) == 1:
        return float(x[0])
    else:
        return tuple([float(v) for v in x])


class Attribute:
    def __init__(
        self,
        init_value: Union[float, tuple[float, ...], np.ndarray],
        value_type: AttributeType,
        range: Optional[tuple[float, float]] = None,
        motion: Optional[Motion] = None,
        function: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    ) -> None:
        np_value = transform_to_numpy(init_value, value_type)
        clipped_value = np.clip(np_value, range[0], range[1]) if range is not None else np_value
        self.init_value: np.ndarray = clipped_value
        self.value_type = value_type
        self.range = range
        self._motion: Optional[Motion] = motion
        self._function: Optional[Callable[[np.ndarray, float], np.ndarray]] = function

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
            motion = Motion(init_value=self.init_value)
            self._motion = motion
        return self._motion

    def disable_motion(self) -> None:
        self._motion = None

    @property
    def motion(self) -> Optional[Motion]:
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
    def function(self) -> Optional[Callable[[np.ndarray, float], np.ndarray]]:
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
