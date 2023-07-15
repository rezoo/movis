import re

import numpy as np
import pandas as pd

from zunda.transform import Transform


class Animation(object):

    timing: str = 'in'

    def __init__(self, start_time: float, end_time: float, scale: float = 1.):
        self.start_time = start_time
        self.end_time = end_time
        self.scale = scale

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __call__(self, time: float) -> Transform:
        if time < self.start_time or self.end_time <= time:
            return Transform()
        p = self.animation_func((time - self.start_time) / self.duration)
        return Transform(
            position=(p.position[0] * self.scale, p.position[1] * self.scale),
            scale=(p.scale[0], p.scale[1]),
            opacity=p.opacity,
        )

    def animation_func(self, t: float) -> Transform:
        return Transform()


class FadeIn(Animation):

    def animation_func(self, t: float) -> Transform:
        return Transform(opacity=t)


class FadeOut(Animation):

    timing: str = 'out'

    def animation_func(self, t: float) -> Transform:
        return Transform(opacity=1. - t)


class BounceUp(Animation):

    def animation_func(self, t: float) -> Transform:
        return Transform(position=(0., - float(np.abs(np.sin(t * np.pi * 2)))))


class HorizontalShake(Animation):

    def animation_func(self, t: float) -> Transform:
        return Transform(position=(np.sin(t * np.pi * 15), 0.))


class VerticalShake(Animation):

    def animation_func(self, t: float) -> Transform:
        return Transform(position=(0., np.sin(t * np.pi * 15)))


def parse_animation_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Animation]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)(?:\s+([\d.e+-]+))?\)'
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
        scale = match.group(4)
        cls = name_to_class[animation_name]
        if cls.timing == 'in':
            s0 = start_time
            s1 = start_time + duration
            assert s1 <= end_time, f'Invalid command: {string}'
        elif cls.timing == 'out':
            s0 = end_time - duration
            s1 = end_time
            assert start_time <= s0, f'Invalid command: {string}'
        else:
            raise ValueError(f'Invalid timing: {cls.timing}')

        if scale is None:
            obj = cls(s0, s1)
        else:
            obj = cls(s0, s1, float(scale))
        animations.append((layer_name, obj))
    return animations


def make_animations_from_timeline(timeline: pd.DataFrame, animation_column: str = 'animation') -> list[tuple[str, Animation]]:
    animations: list[tuple[str, Animation]] = []
    if animation_column not in timeline.columns:
        return animations
    anim_frame = timeline[
        timeline[animation_column].notnull() & (timeline[animation_column] != '')]
    for _, row in anim_frame.iterrows():
        anim_t = parse_animation_command(row['start_time'], row['end_time'], row[animation_column])
        for layer_name, animation in anim_t:
            animations.append((layer_name, animation))
    return animations