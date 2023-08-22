import bisect
import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from .enum import AttributeType, MotionType


def __ease_in_out_medium(t: float):
    alpha = 1.0135673098126083
    beta = 0.0066928509242848554
    return alpha * (1 / (1 + math.exp(- (t - 0.5) / 0.1)) - beta)


def __ease_in_out_expo(t: float):
    alpha = 1.0000908039820193
    beta = 4.5397868702434395e-05
    return alpha * (1 / (1 + math.exp(- (t - 0.5) / 0.05)) - beta)


MOTION_TYPES_TO_FUNC = {
    MotionType.LINEAR: lambda t: t,
    MotionType.EASE_IN: lambda t: t**2,
    MotionType.EASE_OUT: lambda t: 1.0 - (1.0 - t) ** 2,
    MotionType.EASE_IN_OUT: lambda t: t**2 * (3.0 - 2.0 * t),
    MotionType.EASE_IN_MEDIUM: lambda t: t**3,
    MotionType.EASE_OUT_MEDIUM: lambda t: 1.0 - (1.0 - t) ** 3,
    MotionType.EASE_IN_OUT_MEDIUM: __ease_in_out_medium,
    MotionType.EASE_IN_EXPO: lambda t: math.exp(-10.0 * (1 - t)),
    MotionType.EASE_OUT_EXPO: lambda t: 1 - math.exp(-10.0 * t),
    MotionType.EASE_IN_OUT_EXPO: __ease_in_out_expo,
}


class Motion:
    def __init__(
        self,
        init_value: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        value_type: AttributeType = AttributeType.SCALAR,
    ):
        self.keyframes: list[float] = []
        self.values: list[np.ndarray] = []
        self.motion_types: list[Callable[[float], float]] = []
        self.init_value: Optional[np.ndarray] = transform_to_numpy(init_value, value_type) \
            if init_value is not None else None
        self.value_type = value_type

    def __call__(self, prev_value: np.ndarray, layer_time: float) -> np.ndarray:
        if len(self.keyframes) == 0:
            if self.init_value is not None:
                return self.init_value
            raise ValueError("No keyframes")
        elif len(self.keyframes) == 1:
            return self.values[0]

        if layer_time < self.keyframes[0]:
            return self.values[0]
        elif self.keyframes[-1] <= layer_time:
            return self.values[-1]
        else:
            i = bisect.bisect(self.keyframes, layer_time)
            m, M = self.values[i - 1], self.values[i]
            duration = self.keyframes[i] - self.keyframes[i - 1]
            t = (layer_time - self.keyframes[i - 1]) / duration
            t = self.motion_types[i - 1](t)
            return m + (M - m) * t

    def append(
        self,
        keyframe: float,
        value: Union[float, Sequence[float], np.ndarray],
        motion_type: Union[str, MotionType] = MotionType.LINEAR,
    ) -> "Motion":
        i = bisect.bisect(self.keyframes, keyframe)
        self.keyframes.insert(i, float(keyframe))
        self.values.insert(i, np.array(value, dtype=np.float64))
        motion_type = MotionType.from_string(motion_type) \
            if isinstance(motion_type, str) else motion_type
        self.motion_types.insert(i, MOTION_TYPES_TO_FUNC[motion_type])
        return self

    def extend(
        self,
        keyframes: Sequence[float],
        values: Sequence[Union[float, Sequence[float], np.ndarray]],
        motion_types: Optional[Sequence[Union[str, MotionType]]] = None,
    ) -> "Motion":
        assert len(keyframes) == len(values)
        if motion_types is not None:
            if len(motion_types) == len(keyframes) - 1:
                motion_types = list(motion_types) + [MotionType.LINEAR]
            assert len(keyframes) == len(motion_types)
        motion_types = (
            ["linear"] * len(keyframes) if motion_types is None else motion_types
        )
        converted_keyframes = [float(k) for k in keyframes]
        updated_keyframes: list[float] = self.keyframes + converted_keyframes
        converted_values = [
            transform_to_numpy(v, self.value_type) for v in values]
        updated_values: list[np.ndarray[Any, np.dtype[np.float64]]] = self.values + converted_values

        def convert(t: Union[str, MotionType]) -> Callable[[float], float]:
            return MOTION_TYPES_TO_FUNC[MotionType.from_string(t)] \
                if isinstance(t, str) else MOTION_TYPES_TO_FUNC[t]

        converted_motion_types = [convert(t) for t in motion_types]
        updated_motion_types: list[Callable[[float], float]] = self.motion_types + converted_motion_types

        zipped = sorted(zip(updated_keyframes, updated_values, updated_motion_types))
        keyframes_sorted, values_sorted, motion_types_sorted = zip(*zipped)
        self.keyframes = list(keyframes_sorted)
        self.values = list(values_sorted)
        self.motion_types = list(motion_types_sorted)
        return self


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
