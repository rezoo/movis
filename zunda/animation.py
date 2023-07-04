import re
from typing import NamedTuple

import numpy as np


class AnimationProperty(NamedTuple):

    position: tuple[float, float] = (0., 0.)
    scale: tuple[float, float] = (1., 1.)
    opacity: float = 1.


class Animation(object):

    def __init__(self, start_time: float, end_time: float, scale: float = 1.):
        self.start_time = start_time
        self.end_time = end_time
        self.scale = scale

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __call__(self, time: float) -> AnimationProperty:
        if time < self.start_time or self.end_time <= time:
            return AnimationProperty()
        p = self.animation_func((time - self.start_time) / self.duration)
        return AnimationProperty(
            position=(p.position[0] * self.scale, p.position[1] * self.scale),
            scale=(p.scale[0], p.scale[1]),
            opacity=p.opacity,
        )

    def animation_func(self, t: float) -> AnimationProperty:
        return AnimationProperty()


class BounceUp(Animation):

    def animation_func(self, t: float) -> AnimationProperty:
        return AnimationProperty(position=(0., - float(np.abs(np.sin(t * np.pi * 2)))))


class HorizontalShake(Animation):

    def animation_func(self, t: float) -> AnimationProperty:
        return AnimationProperty(position=(np.sin(t * np.pi * 15), 0.))


class VerticalShake(Animation):

    def animation_func(self, t: float) -> AnimationProperty:
        return AnimationProperty(position=(0., np.sin(t * np.pi * 15)))


def parse_animation_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Animation]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)\s+([\d.e+-]+)\)'
    name_to_class = {
        'BounceUp': BounceUp,
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
