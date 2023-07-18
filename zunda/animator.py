import re
from typing import Optional

import pandas as pd

from zunda.layer import Composition


class Animator(object):

    def __init__(self, start_time: float, end_time: float, duration: float, scale: Optional[float] = None):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.scale = scale

    def __call__(self, scene: Composition, layer_name: str) -> None:
        raise NotImplementedError


class FadeIn(Animator):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        layer = scene.get_layer(layer_name)
        motion = scene.get_motion(layer.name, 'opacity', auto_create=True)
        value = float(motion(self.start_time + self.duration))
        motion.append(self.start_time, 0.0, 'linear')
        motion.append(self.start_time + self.duration, value)


class FadeOut(Animator):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        layer = scene.get_layer(layer_name)
        motion = scene.get_motion(layer.name, 'opacity', auto_create=True)
        value = float(motion(self.end_time - self.duration))
        motion.append(self.end_time - self.duration, value, 'linear')
        motion.append(self.end_time, 0.0)


class BounceUp(Animator):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        layer = scene.get_layer(layer_name)
        motion = scene.get_motion(layer.name, 'position', auto_create=True)
        p0 = motion(self.start_time)
        p1 = (p0[0], p0[1] - self.scale)
        t0, T = self.start_time, self.duration
        motion.append(t0, p0, 'ease_out')
        motion.append(t0 + T * 0.25, p1, 'ease_in')
        motion.append(t0 + T * 0.50, p0, 'ease_out')
        motion.append(t0 + T * 0.75, p1, 'ease_in')
        motion.append(t0 + T, p0)


def parse_animation_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Animator]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)(?:\s+([\d.e+-]+))?\)'
    name_to_class = {
        'FadeIn': FadeIn,
        'FadeOut': FadeOut,
        'BounceUp': BounceUp,
    }
    animations = []
    for string in command.split(';'):
        match = re.match(pattern, string)
        assert match is not None, f'Invalid command: {string}'
        layer_name = match.group(1)
        animation_name = match.group(2)
        duration = float(match.group(3))
        scale = float(match.group(4)) if match.group(4) is not None else None
        cls = name_to_class[animation_name]
        animator = cls(start_time, end_time, duration, scale)
        animations.append((layer_name, animator))
    return animations


def make_animations_from_timeline(timeline: pd.DataFrame, animation_column: str = 'animation') -> list[tuple[str, Animator]]:
    animations: list[tuple[str, Animator]] = []
    if animation_column not in timeline.columns:
        return animations
    anim_frame = timeline[
        timeline[animation_column].notnull() & (timeline[animation_column] != '')]
    for _, row in anim_frame.iterrows():
        anim_t = parse_animation_command(row['start_time'], row['end_time'], row[animation_column])
        for layer_name, animation in anim_t:
            animations.append((layer_name, animation))
    return animations
