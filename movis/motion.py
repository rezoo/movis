from __future__ import annotations

import bisect
from typing import Any, Callable, Sequence

import numpy as np

from .enum import AttributeType, Easing


def linear(t: float) -> float:
    return t


def flat(t: float) -> float:
    return 0.0


class EaseIn:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, t: float) -> float:
        return t ** self.n


class EaseOut:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, t: float) -> float:
        return 1.0 - (1.0 - t) ** self.n


class EaseInOut:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, t: float) -> float:
        return 0.5 * (2 * t) ** self.n if t < 0.5 else 1.0 - 0.5 * (1.0 - 2 * (t - 0.5)) ** self.n


EASING_TO_FUNC: dict[Easing, Callable[[float], float]] = {  # type: ignore
    Easing.LINEAR: linear,
    Easing.EASE_IN: EaseIn(2),
    Easing.EASE_OUT: EaseOut(2),
    Easing.EASE_IN_OUT: EaseInOut(2),
    Easing.FLAT: flat,
    Easing.EASE_IN2: EaseIn(2),
    Easing.EASE_IN3: EaseIn(3),
    Easing.EASE_IN4: EaseIn(4),
    Easing.EASE_IN5: EaseIn(5),
    Easing.EASE_IN6: EaseIn(6),
    Easing.EASE_IN7: EaseIn(7),
    Easing.EASE_IN8: EaseIn(8),
    Easing.EASE_IN9: EaseIn(9),
    Easing.EASE_IN10: EaseIn(10),
    Easing.EASE_IN12: EaseIn(12),
    Easing.EASE_IN14: EaseIn(14),
    Easing.EASE_IN16: EaseIn(16),
    Easing.EASE_IN18: EaseIn(18),
    Easing.EASE_IN20: EaseIn(20),
    Easing.EASE_IN25: EaseIn(25),
    Easing.EASE_IN30: EaseIn(30),
    Easing.EASE_IN35: EaseIn(35),
    Easing.EASE_OUT2: EaseOut(2),
    Easing.EASE_OUT3: EaseOut(3),
    Easing.EASE_OUT4: EaseOut(4),
    Easing.EASE_OUT5: EaseOut(5),
    Easing.EASE_OUT6: EaseOut(6),
    Easing.EASE_OUT7: EaseOut(7),
    Easing.EASE_OUT8: EaseOut(8),
    Easing.EASE_OUT9: EaseOut(9),
    Easing.EASE_OUT10: EaseOut(10),
    Easing.EASE_OUT12: EaseOut(12),
    Easing.EASE_OUT14: EaseOut(14),
    Easing.EASE_OUT16: EaseOut(16),
    Easing.EASE_OUT18: EaseOut(18),
    Easing.EASE_OUT20: EaseOut(20),
    Easing.EASE_OUT25: EaseOut(25),
    Easing.EASE_OUT30: EaseOut(30),
    Easing.EASE_OUT35: EaseOut(35),
    Easing.EASE_IN_OUT2: EaseInOut(2),
    Easing.EASE_IN_OUT3: EaseInOut(3),
    Easing.EASE_IN_OUT4: EaseInOut(4),
    Easing.EASE_IN_OUT5: EaseInOut(5),
    Easing.EASE_IN_OUT6: EaseInOut(6),
    Easing.EASE_IN_OUT7: EaseInOut(7),
    Easing.EASE_IN_OUT8: EaseInOut(8),
    Easing.EASE_IN_OUT9: EaseInOut(9),
    Easing.EASE_IN_OUT10: EaseInOut(10),
    Easing.EASE_IN_OUT12: EaseInOut(12),
    Easing.EASE_IN_OUT14: EaseInOut(14),
    Easing.EASE_IN_OUT16: EaseInOut(16),
    Easing.EASE_IN_OUT18: EaseInOut(18),
    Easing.EASE_IN_OUT20: EaseInOut(20),
    Easing.EASE_IN_OUT25: EaseInOut(25),
    Easing.EASE_IN_OUT30: EaseInOut(30),
    Easing.EASE_IN_OUT35: EaseInOut(35),
}


class Motion:
    """Defines a motion of ``Attribute`` instances used for keyframe animations.

    This instance is basically initialized by calling ``enable_motion()`` in ``Attribute`` instances.
    keyframes, values and correspoing motion types can be added using ``append()`` or ``extend()``.
    By using these keyframes, ``Motion`` instance returns the complemented values at the given time.

    Args:
        init_value:
            The initial value to use. It is used when no animation is set,
            or when an animation with zero keyframes is set.
        value_type:
            Specifies the type of value that the ``Attribute`` will handle.

    Examples:
        >>> import movis as mv
        >>> attr = mv.Attribute(1.0, mv.AttributeType.SCALAR)
        >>> attr.enable_motion().append(0.0, 0.0).append(1.0, 1.0)
        >>> len(attr.motion)
        2
        >>> attr.enable_motion().extend([2.0, 3.0], [2.0, 3.0], ['ease_in_out'])
        >>> len(attr.motion)
        4
        >>> attr.motion.clear()
        >>> len(attr.motion)
        0
        >>> assert attr(0.0) == attr.init_value
    """

    def __init__(
        self,
        init_value: float | Sequence[float] | np.ndarray | None = None,
        value_type: AttributeType = AttributeType.SCALAR,
    ):
        self.keyframes: list[float] = []
        self.values: list[np.ndarray] = []
        self.easings: list[Callable[[float], float]] = []
        self.init_value: np.ndarray | None = transform_to_numpy(init_value, value_type) \
            if init_value is not None else None
        self.value_type = value_type

    def __len__(self) -> int:
        return len(self.keyframes)

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
            t = self.easings[i - 1](t)
            return m + (M - m) * t

    def append(
        self,
        keyframe: float,
        value: float | Sequence[float] | np.ndarray,
        easing: str | Easing | Callable[[float], float] = Easing.LINEAR,
    ) -> "Motion":
        """Append a single keyframe.

        Args:
            keyframe:
                time of the keyframe.
            value:
                value of the keyframe.
            easing:
                motion type of the keyframe. This must be a string,
                ``Easing`` enum, or an easing function
                ``f: float -> float`` that satisfies ``f(0) == 0`` and ``f(1) == 1``.
                The default is ``Easing.LINEAR`` (linear completion).
        """
        i = bisect.bisect(self.keyframes, keyframe)
        if 0 < i and self.keyframes[i - 1] == keyframe:
            raise ValueError(f"Keyframe {keyframe} already exists")
        self.keyframes.insert(i, float(keyframe))
        self.values.insert(i, transform_to_numpy(value, self.value_type))
        if isinstance(easing, str):
            easing_func = EASING_TO_FUNC[Easing.from_string(easing)]
        elif isinstance(easing, Easing):
            easing_func = EASING_TO_FUNC[easing]
        elif callable(easing):
            easing_func = easing
        else:
            raise ValueError(f"Invalid easing type: {type(easing)}")
        self.easings.insert(i, easing_func)
        return self

    def extend(
        self,
        keyframes: Sequence[float],
        values: Sequence[float | Sequence[float] | np.ndarray],
        easings: Sequence[str | Easing | Callable[[float], float]] | None = None,
    ) -> "Motion":
        """Append multiple keyframes.

        Args:
            keyframes:
                times of the keyframes.
            values:
                values of the keyframes. The length of ``values`` must be the same as ``keyframes``.
            easings:
                motion types of the keyframes. Each element of ``easings`` must be a string, ``Easing`` enum,
                or an easing function ``f: float -> float`` that satisfies ``f(0) == 0`` and ``f(1) == 1``.
                Note that the length of ``easings`` must be the same as ``len(keyframes)`` or ``len(keyframes) - 1``.
                If ``easings`` is ``None``, ``Easing.LINEAR`` is used for all keyframes.
                If ``len(easings) == len(keyframes) - 1``, ``Motion`` automatically adds
                ``Easing.LINEAR`` to the end of ``easings``.

        Examples:
            >>> import movis as mv
            >>> motion = mv.Motion(value_type=mv.AttributeType.SCALAR)
            >>> # add two keyframes and The type of motion for that period is ``Easing.EASE_IN_OUT``
            >>> motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=['ease_in_out'])
            >>> # add other two keyframes and The type of motion for that period is ``Easing.LINEAR``
            >>> motion.extend([2.0, 3.0], [2.0, 3.0])
            >>> len(motion)
            4
        """
        assert len(keyframes) == len(values)
        if isinstance(easings, str) or isinstance(easings, Easing):
            raise ValueError("easings must be a list of strings or Easing enums")
        if easings is not None:
            if len(easings) == len(keyframes) - 1:
                easings = list(easings) + [Easing.LINEAR]
            assert len(keyframes) == len(easings)
        easings = (
            ["linear"] * len(keyframes) if easings is None else easings
        )
        converted_keyframes = [float(k) for k in keyframes]

        # Check if given keyframes already exist
        seen = set(self.keyframes)
        for keyframe in converted_keyframes:
            if keyframe in seen:
                raise ValueError(f"Keyframe {keyframe} already exists")
            seen.add(keyframe)

        updated_keyframes: list[float] = self.keyframes + converted_keyframes
        converted_values = [
            transform_to_numpy(v, self.value_type) for v in values]
        updated_values: list[np.ndarray[Any, np.dtype[np.float64]]] = self.values + converted_values

        def convert(t: str | Easing | Callable[[float], float]) -> Callable[[float], float]:
            if callable(t):
                return t
            elif isinstance(t, Easing):
                return EASING_TO_FUNC[t]
            elif isinstance(t, str):
                return EASING_TO_FUNC[Easing.from_string(t)]
            else:
                raise ValueError(f"Invalid easing type: {type(t)}")

        converted_easings = [convert(t) for t in easings]
        updated_easings: list[Callable[[float], float]] = self.easings + converted_easings

        zipped = sorted(zip(updated_keyframes, updated_values, updated_easings))
        keyframes_sorted, values_sorted, easings_sorted = zip(*zipped)
        self.keyframes = list(keyframes_sorted)
        self.values = list(values_sorted)
        self.easings = list(easings_sorted)
        return self

    def clear(self) -> "Motion":
        self.keyframes = []
        self.values = []
        self.easings = []
        return self


def transform_to_numpy(value: int | float | Sequence[float] | np.ndarray, value_type: AttributeType) -> np.ndarray:
    """Transform a scalar, tuple, list or numpy array to a numpy array.

    Args:
        value:
            The value to transform.
        value_type:
            The type of value that the attribute handles.

    Returns:
        A numpy array with the length of 1, 2 or 3.
        This length is determined by the ``value_type``.

    Examples:
        >>> from movis.motion import transform_to_numpy
        >>> transform_to_numpy(1.0, mv.AttributeType.SCALAR)
        array([1.])
        >>> transform_to_numpy([1.0, 2.0], mv.AttributeType.VECTOR2D)
        array([1., 2.])
        >>> transform_to_numpy(1.0, mv.AttributeType.VECTOR3D)
        array([1., 1., 1.])
    """
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
