import bisect
from typing import Optional, TypeVar, Generic, Tuple, Callable


T = TypeVar('T', float, Tuple[float, float])


class Motion(Generic[T]):

    def __call__(self, time: float) -> T:
        raise NotImplementedError

    def append(self, keyframe: float, value: T, type: str = 'linear') -> None:
        raise NotImplementedError


motion_types_to_func = {
    'linear': lambda t: t,
    'ease_in': lambda t: t ** 2,
    'ease_out': lambda t: 1. - (1. - t) ** 2,
    'ease_in_out': lambda t: t ** 2 * (3. - 2. * t),
}


class MotionSequence(Motion[T]):

    def __init__(self, default_value: Optional[T] = None):
        self.keyframes: list[float] = []
        self.values: list[T] = []
        self.motion_types: list[Callable[[float], float]] = []
        self.default_value: Optional[T] = default_value

    def __call__(self, time: float) -> T:
        if len(self.keyframes) == 0:
            if self.default_value is not None:
                return self.default_value
            raise ValueError('No keyframes')
        elif len(self.keyframes) == 1:
            return self.values[0]

        if time < self.keyframes[0]:
            return self.values[0]
        elif self.keyframes[-1] <= time:
            return self.values[-1]
        else:
            i = bisect.bisect(self.keyframes, time)
            m, M = self.values[i - 1], self.values[i]
            duration = self.keyframes[i] - self.keyframes[i - 1]
            t = (time - self.keyframes[i - 1]) / duration
            t = self.motion_types[i - 1](t)
            if isinstance(m, float) and isinstance(M, float):
                return float(m + (M - m) * t)
            elif isinstance(m, tuple) and isinstance(M, tuple):
                return (m[0] + (M[0] - m[0]) * t, m[1] + (M[1] - m[1]) * t)
            else:
                raise ValueError(f'Unexpected value: {m}, {M}')

    def append(self, keyframe: float, value: T, type: str = 'linear') -> None:
        i = bisect.bisect(self.keyframes, keyframe)
        self.keyframes.insert(i, keyframe)
        self.values.insert(i, value)
        self.motion_types.insert(i, motion_types_to_func[type])

    def extend(self, keyframes: list[float], values: list[T], types: Optional[list[str]] = None) -> None:
        if types is None:
            types = ['linear'] * len(keyframes)
        for keyframe, value, type in zip(keyframes, values, types):
            self.append(keyframe, value, type)
