import bisect
import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from .enum import MotionType

MOTION_TYPES_TO_FUNC = {
    MotionType.LINEAR: lambda t: t,
    MotionType.EASE_IN: lambda t: t**2,
    MotionType.EASE_OUT: lambda t: 1.0 - (1.0 - t) ** 2,
    MotionType.EASE_IN_OUT: lambda t: t**2 * (3.0 - 2.0 * t),
    MotionType.EASE_IN_MEDIUM: lambda t: t**3,
    MotionType.EASE_OUT_MEDIUM: lambda t: 1.0 - (1.0 - t) ** 3,
    MotionType.EASE_IN_OUT_MEDIUM: lambda t: 1 / (1 + math.exp(- (t - 0.5) / 0.1)),
    MotionType.EASE_IN_EXPO: lambda t: math.exp(-10.0 * (1 - t)),
    MotionType.EASE_OUT_EXPO: lambda t: 1 - math.exp(-10.0 * t),
    MotionType.EASE_IN_OUT_EXPO: lambda t: 1 / (1 + math.exp(- (t - 0.5) / 0.05))
}


class Motion:
    def __init__(
        self,
        init_value: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ):
        self.keyframes: list[float] = []
        self.values: list[np.ndarray] = []
        self.motion_types: list[Callable[[float], float]] = []
        self.init_value: Optional[np.ndarray] = (
            np.array(init_value, dtype=np.float64)
            if init_value is not None
            else None
        )

    def __call__(self, layer_time: float, prev_value: np.ndarray) -> np.ndarray:
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
            assert len(keyframes) == len(motion_types)
        motion_types = (
            ["linear"] * len(keyframes) if motion_types is None else motion_types
        )
        converted_keyframes = [float(k) for k in keyframes]
        updated_keyframes: list[float] = self.keyframes + converted_keyframes
        converted_values = [np.array(v, dtype=np.float64) for v in values]
        updated_values: list[np.ndarray[Any, np.dtype[np.float64]]] = self.values + converted_values

        def convert(t: Union[str, MotionType]) -> Callable[[float], float]:
            return MOTION_TYPES_TO_FUNC[MotionType.from_string(t)] \
                if isinstance(t, str) else MOTION_TYPES_TO_FUNC[t]

        converted_motion_types = [convert(t) for t in motion_types]
        updated_motion_types: list[Callable[[float], float]] = self.motion_types + converted_motion_types

        zipped = sorted(zip(updated_keyframes, updated_values, updated_motion_types))
        keyframes_sorted, values_sorted, motion_types_sorted = zip(*zipped)
        self.keyframes = keyframes_sorted
        self.values = values_sorted
        self.motion_types = motion_types_sorted
        return self
