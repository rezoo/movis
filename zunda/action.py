import re
from typing import Optional, Sequence

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


def make_action_functions(
        start_time: Sequence[float], end_time: Sequence[float], actions: Sequence[str]) -> list[tuple[str, Action]]:
    assert len(start_time) == len(end_time) == len(actions)
    animations: list[tuple[str, Action]] = []
    start_time = [t for (t, a) in zip(start_time, actions) if isinstance(a, str) and a != '']
    end_time = [t for (t, a) in zip(end_time, actions) if isinstance(a, str) and a != '']
    actions = [a for a in actions if isinstance(a, str) and a != '']
    for t0, t1, action_str in zip(start_time, end_time, actions):
        actions_t = parse_action_command(t0, t1, action_str)
        for layer_name, action_func in actions_t:
            animations.append((layer_name, action_func))
    return animations
