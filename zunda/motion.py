import bisect
import math
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

motion_types_to_func = {
    "linear": lambda t: t,
    "ease_in": lambda t: t**2,
    "ease_out": lambda t: 1.0 - (1.0 - t) ** 2,
    "ease_in_out": lambda t: t**2 * (3.0 - 2.0 * t),
    "ease_in_cubic": lambda t: t**3,
    "ease_out_cubic": lambda t: 1.0 - (1.0 - t) ** 3,
    "ease_in_expo": lambda t: math.exp(-10.0 * (1 - t)),
    "ease_out_expo": lambda t: 1 - math.exp(-10.0 * t),
}


class Motion:
    def __init__(
        self,
        default_value: Optional[Union[float, Sequence[float], np.ndarray[Any, Any]]] = None,
    ):
        self.keyframes: list[float] = []
        self.values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        self.motion_types: list[Callable[[float], float]] = []
        self.default_value: Optional[np.ndarray[Any, np.dtype[np.float64]]] = (
            np.array(default_value, dtype=np.float64)
            if default_value is not None
            else None
        )

    def __call__(self, layer_time: float) -> np.ndarray[Any, np.dtype[np.float64]]:
        if len(self.keyframes) == 0:
            if self.default_value is not None:
                return self.default_value
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
        value: Union[float, Sequence[float], np.ndarray[Any, Any]],
        motion_type: str = "linear",
    ) -> "Motion":
        i = bisect.bisect(self.keyframes, keyframe)
        self.keyframes.insert(i, float(keyframe))
        self.values.insert(i, np.array(value, dtype=np.float64))
        self.motion_types.insert(i, motion_types_to_func[motion_type])
        return self

    def extend(
        self,
        keyframes: Sequence[float],
        values: Sequence[Union[float, Sequence[float], np.ndarray[Any, Any]]],
        motion_types: Optional[Sequence[str]] = None,
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
        converted_motion_types = [motion_types_to_func[t] for t in motion_types]
        updated_motion_types: list[Callable[[float], float]] = self.motion_types + converted_motion_types

        zipped = sorted(zip(updated_keyframes, updated_values, updated_motion_types))
        keyframes_sorted, values_sorted, motion_types_sorted = zip(*zipped)
        self.keyframes = keyframes_sorted
        self.values = values_sorted
        self.motion_types = motion_types_sorted
        return self
