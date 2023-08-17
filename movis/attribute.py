from __future__ import annotations

from typing import Callable, Hashable, Sequence, Union

import numpy as np

from movis.enum import AttributeType
from movis.motion import Motion


def normalize_to_numpy(value: Union[int, float, Sequence[float], np.ndarray], value_type: AttributeType) -> np.ndarray:
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
    def __init__(self, init_value: Union[float, tuple[float, ...], np.ndarray], value_type: AttributeType):
        self.init_value: np.ndarray = normalize_to_numpy(init_value, value_type)
        self.value_type = value_type
        self._motions: list[Callable[[float, np.ndarray], np.ndarray]] = []

    def __call__(self, layer_time: float) -> np.ndarray:
        if len(self._motions) == 0:
            return self.init_value
        else:
            value = self.init_value
            for motion in self._motions:
                value = motion(layer_time, value)
            return value

    def has_motion(self) -> bool:
        return 0 < len(self._motions)

    def append(self, motion: Callable[[float, np.ndarray], np.ndarray]) -> None:
        self._motions.append(motion)

    def enable_animation(self) -> Motion:
        motions = [m for m in self._motions if isinstance(m, Motion)]
        if 0 < len(motions):
            return motions[-1]
        else:
            motion = Motion(init_value=self.init_value)
            self.append(motion)
            return motion

    def __repr__(self) -> str:
        if len(self._motions) == 0:
            return f"{self.init_value}"
        else:
            return f"Attribute(value_type={self.value_type})"


class AttributesMixin:
    @property
    def attributes(self) -> dict[str, Attribute]:
        return {key: attr for key, attr in vars(self).items() if isinstance(attr, Attribute)}

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        return tuple([transform_to_hashable(attr(time)) for attr in self.attributes.values()])
