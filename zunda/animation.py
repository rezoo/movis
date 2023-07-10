import re

import numpy as np

from zunda.transform import TransformProperty


class Animation(object):

    def __init__(self, start_time: float, end_time: float, scale: float = 1.):
        self.start_time = start_time
        self.end_time = end_time
        self.scale = scale

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __call__(self, time: float) -> TransformProperty:
        if time < self.start_time or self.end_time <= time:
            return TransformProperty()
        p = self.animation_func((time - self.start_time) / self.duration)
        return TransformProperty(
            position=(p.position[0] * self.scale, p.position[1] * self.scale),
            scale=(p.scale[0], p.scale[1]),
            opacity=p.opacity,
        )

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty()


class FadeIn(Animation):

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty(opacity=t)


class FadeOut(Animation):

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty(opacity=1. - t)


class BounceUp(Animation):

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty(position=(0., - float(np.abs(np.sin(t * np.pi * 2)))))


class HorizontalShake(Animation):

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty(position=(np.sin(t * np.pi * 15), 0.))


class VerticalShake(Animation):

    def animation_func(self, t: float) -> TransformProperty:
        return TransformProperty(position=(0., np.sin(t * np.pi * 15)))


def parse_animation_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Animation]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)\s+([\d.e+-]+)\)'
    name_to_class = {
        'FadeIn': FadeIn,
        'FadeOut': FadeOut,
        'BounceUp': BounceUp,
        'HorizontalShake': HorizontalShake,
        'VerticalShake': VerticalShake,
    }
    animations = []
    for string in command.split(';'):
        match = re.match(pattern, string)
        assert match is not None, f'Invalid command: {string}'
        layer_name = match.group(1)
        animation_name = match.group(2)
        duration = float(match.group(3))
        scale = float(match.group(4))
        cls = name_to_class[animation_name]
        obj = cls(start_time, start_time + duration, scale)
        animations.append((layer_name, obj))
    return animations
