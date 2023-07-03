import re

import numpy as np


class Animation(object):

    def __init__(self, start_time: float, end_time: float, scale: float = 1.0):
        self.start_time = start_time
        self.end_time = end_time
        self.scale = scale

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __call__(self, time: float) -> tuple[float, float]:
        if time < self.start_time or self.end_time <= time:
            return (0., 0.)
        vec = self.position_func((time - self.start_time) / self.duration)
        return (vec[0] * self.scale, vec[1] * self.scale)

    def position_func(self, t: float) -> tuple[float, float]:
        return [1., 1.]


class BounceUp(Animation):

    def position_func(self, t: float) -> float:
        return (0., - float(np.abs(np.sin(t * np.pi * 2))))


class BounceDown(Animation):

    def position_func(self, t: float) -> float:
        return (0., float(np.abs(np.sin(t * np.pi * 2))))


class HorizontalShake(Animation):

    def position_func(self, t: float) -> tuple[float, float]:
        return (np.sin(t * np.pi * 15), 0.)


class VerticalShake(Animation):

    def position_func(self, t: float) -> tuple[float, float]:
        return (0., np.sin(t * np.pi * 15))


def parse_animation_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Animation]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)\s+([\d.e+-]+)\)'
    name_to_class = {
        'BounceUp': BounceUp,
        'BounceDown': BounceDown,
        'HorizontalShake': HorizontalShake,
        'VerticalShake': VerticalShake,
    }
    animations = []
    for string in command.split(';'):
        match = re.match(pattern, string)
        layer_name = match.group(1)
        animation_name = match.group(2)
        duration = float(match.group(3))
        scale = float(match.group(4))
        cls = name_to_class[animation_name]
        obj = cls(start_time, start_time + duration, scale)
        animations.append((layer_name, obj))
    return animations
