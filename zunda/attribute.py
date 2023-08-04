from __future__ import annotations

from enum import Enum
from typing import Callable, Sequence, Union

import numpy as np

from zunda.motion import Motion


class AttributeType(Enum):
    ANY = -1
    SCALAR = 0
    VECTOR2D = 1
    VECTOR3D = 2
    ANGLE = 3
    COLOR = 4

    @staticmethod
    def from_string(s: str) -> "AttributeType":
        if s == 'scalar':
            return AttributeType.SCALAR
        elif s == 'vector2d':
            return AttributeType.VECTOR2D
        elif s == 'vector3d':
            return AttributeType.VECTOR3D
        elif s == 'angle':
            return AttributeType.ANGLE
        else:
            raise ValueError(f"Unknown attribute type: {s}")


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


def normalize_to_1dscalar(x: Union[float, Sequence[float]]) -> float:
    if isinstance(x, float):
        return x
    elif len(x) == 1:
        return float(x[0])
    else:
        raise ValueError(f"Invalid value: {x}")


def normalize_to_2dvector(
    x: Union[float, Sequence[float], np.ndarray]
) -> tuple[float, float]:
    if isinstance(x, float):
        return (x, x)
    elif len(x) == 1:
        return (float(x[0]), float(x[0]))
    else:
        return (float(x[0]), float(x[1]))


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

    def enable(self) -> Motion:
        motions = [m for m in self._motions if isinstance(m, Motion)]
        if 0 < len(motions):
            return motions[-1]
        else:
            motion = Motion(init_value=self.init_value)
            self.append(motion)
            return motion
