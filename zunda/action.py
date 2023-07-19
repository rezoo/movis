import re
from typing import Optional

import pandas as pd

from zunda.layer import Composition


class Action(object):

    def __init__(self, start_time: float, end_time: float, duration: float, scale: Optional[float] = None):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.scale = scale

    def __call__(self, scene: Composition, layer_name: str) -> None:
        raise NotImplementedError


class FadeIn(Action):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        motion = scene.enable_motion(layer_name, 'opacity')
        value = float(motion(self.start_time + self.duration))
        motion.append(self.start_time, 0.0, 'linear')
        motion.append(self.start_time + self.duration, value)


class FadeOut(Action):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        motion = scene.enable_motion(layer_name, 'opacity')
        value = float(motion(self.end_time - self.duration))
        motion.append(self.end_time - self.duration, value, 'linear')
        motion.append(self.end_time, 0.0)


class BounceUp(Action):

    def __call__(self, scene: Composition, layer_name: str) -> None:
        motion = scene.enable_motion(layer_name, 'position')
        p0 = motion(self.start_time)
        p1 = (p0[0], p0[1] - self.scale)
        t0, T = self.start_time, self.duration
        motion.append(t0, p0, 'ease_out')
        motion.append(t0 + T * 0.25, p1, 'ease_in')
        motion.append(t0 + T * 0.50, p0, 'ease_out')
        motion.append(t0 + T * 0.75, p1, 'ease_in')
        motion.append(t0 + T, p0)


def parse_action_command(
        start_time: float, end_time: float, command: str) -> list[tuple[str, Action]]:
    pattern = r'(\w+)\.(\w+)\(([\d.e+-]+)(?:\s+([\d.e+-]+))?\)'
    name_to_class = {
        'FadeIn': FadeIn,
        'FadeOut': FadeOut,
        'BounceUp': BounceUp,
    }
    actions = []
    for string in command.split(';'):
        match = re.match(pattern, string)
        assert match is not None, f'Invalid command: {string}'
        layer_name = match.group(1)
        animation_name = match.group(2)
        duration = float(match.group(3))
        scale = float(match.group(4)) if match.group(4) is not None else None
        cls = name_to_class[animation_name]
        action_func = cls(start_time, end_time, duration, scale)
        actions.append((layer_name, action_func))
    return actions


def make_action_functions_from_timeline(timeline: pd.DataFrame, action_column: str = 'action') -> list[tuple[str, Action]]:
    animations: list[tuple[str, Action]] = []
    if action_column not in timeline.columns:
        return animations
    anim_frame = timeline[
        timeline[action_column].notnull() & (timeline[action_column] != '')]
    for _, row in anim_frame.iterrows():
        anim_t = parse_action_command(row['start_time'], row['end_time'], row[action_column])
        for layer_name, action_func in anim_t:
            animations.append((layer_name, action_func))
    return animations
