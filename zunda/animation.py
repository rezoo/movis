import re
from typing import Optional, TypeVar, Union, Generic, Tuple

import numpy as np
import pandas as pd

from zunda.transform import Transform


T = TypeVar('T', float, Tuple[float, float])


class SingleAnimation(Generic[T]):

    def __init__(self, time_range: tuple[float, float], value_range: tuple[T, T], type: str = 'linear'):
        self.time_range = time_range
        self.value_range: tuple[T, T] = value_range
        self.type = type
        if type != 'linear':
            raise NotImplementedError

    @property
    def duration(self) -> float:
        return self.time_range[1] - self.time_range[0]

    def __call__(self, time: float) -> T:
        m, M = self.value_range
        if time < self.time_range[0]:
            return m
        elif self.time_range[1] <= time:
            return M
        p = (time - self.time_range[0]) / self.duration

        if isinstance(m, float) and isinstance(M, float):
            return float(m + (M - m) * p)
        elif isinstance(m, tuple) and isinstance(M, tuple):
            return (m[0] + (M[0] - m[0]) * p, m[1] + (M[1] - m[1]) * p)
        else:
            raise ValueError(f'Unexpected value: {m}, {M}')


class AnimationSequence(Generic[T]):

    def __init__(self, timeline: pd.DataFrame):
        if 'type' not in timeline.columns:
            timeline['type'] = 'linear'
        frame = pd.DataFrame({
            'start_time': timeline['time'][:-1].values,
            'end_time': timeline['time'][1:].values,
        })
        if 'value' in timeline.columns:
            value_type = 'scalar'
            frame['start_value'] = timeline['value'][:-1].values
            frame['end_value'] = timeline['value'][1:].values
        elif set(['value_x', 'value_y']).issubset(timeline.columns):
            value_type = '2dvector'
            frame['start_value_x'] = timeline['value_x'][:-1].values
            frame['start_value_y'] = timeline['value_y'][:-1].values
            frame['end_value_x'] = timeline['value_x'][1:].values
            frame['end_value_y'] = timeline['value_y'][1:].values
        else:
            raise ValueError(f'{timeline.columns} do not contain (start_value, end_value) '
                             'or (start_value_x, end_value_x, start_value_y, end_value_y))')
        self.timeline = frame
        self.animations: list[SingleAnimation[T]] = []
        for _, row in self.timeline.iterrows():
            time_range = (row['start_time'], row['end_time'])
            if value_type == 'scalar':
                value_range = (row['start_value'], row['end_value'])
            elif value_type == '2dvector':
                value_range = (
                    (row['start_value_x'], row['start_value_y']),
                    (row['end_value_x'], row['end_value_y']))
            else:
                raise ValueError(f'Unexpected value: {row}')
            self.animations.append(SingleAnimation(time_range, value_range, type=row['type']))
        self.time_range = (self.timeline['start_time'].min(), self.timeline['end_time'].max())

    @property
    def duration(self) -> float:
        return self.time_range[1] - self.time_range[0]

    def get_state(self, time: float) -> Optional[pd.Series]:
        idx = self.timeline['start_time'].searchsorted(time, side='right') - 1
        if idx >= 0 and self.timeline['end_time'].iloc[idx] > time:
            return self.timeline.iloc[idx]
        else:
            return None

    def __call__(self, time: float) -> T:
        if time < self.time_range[0]:
            return self.animations[0].value_range[0]
        elif self.time_range[1] <= time:
            return self.animations[-1].value_range[1]
        else:
            row = self.get_state(time)
            assert row is not None, f'Unexpected error: {time}'
            animation = self.animations[row.index]
            return animation(time)


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
