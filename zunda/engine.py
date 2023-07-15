from collections import defaultdict
import os
from typing import Optional, NamedTuple, Iterable, Any, DefaultDict

from cachetools import LRUCache
import ffmpeg
import imageio
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from zunda.animation import Animation, parse_animation_command
from zunda.utils import get_voicevox_dataframe, rand_from_string, normalize_2dvector
from zunda.transform import Transform, resize, alpha_composite


class Layer(object):

    def __init__(self, timeline: pd.DataFrame) -> None:
        self.timeline = timeline

    def get_state(self, time: float) -> Optional[pd.Series]:
        idx = self.timeline['start_time'].searchsorted(time, side='right') - 1
        if idx >= 0 and self.timeline['end_time'].iloc[idx] > time:
            return self.timeline.iloc[idx]
        else:
            return None

    @property
    def start_time(self):
        return self.timeline['start_time'].min()

    @property
    def end_time(self):
        return self.timeline['end_time'].max()

    @property
    def duration(self):
        return self.timeline['end_time'].max() - self.timeline['start_time'].min()

    def get_keys(self, time: float) -> tuple[Any, ...]:
        raise NotImplementedError

    def render(self, time: float) -> Image.Image:
        raise NotImplementedError


class ImageLayer(Layer):

    def __init__(self, timeline: pd.DataFrame, img_path: str) -> None:
        super().__init__(timeline)
        self.image = Image.open(img_path).convert('RGBA')

    def get_keys(self, time: float) -> tuple[int]:
        state = self.get_state(time)
        key = 1 if state is not None else 0
        return (key,)

    def render(self, time: float) -> Image.Image:
        return self.image


class SlideLayer(Layer):

    def __init__(
            self, timeline: pd.DataFrame,
            slide_path: str, slide_column: str = 'slide') -> None:
        super().__init__(timeline)
        self.slide_timeline = np.cumsum(self.timeline[slide_column])
        slide_images = []
        for img in convert_from_path(slide_path):
            img = img.convert('RGBA')
            slide_images.append(img)
        self.slide_images = slide_images

    def get_keys(self, time: float) -> tuple[Any, ...]:
        state = self.get_state(time)
        key = int(self.slide_timeline[state.name]) if state is not None else None
        return (key,)

    def render(self, time: float) -> Image.Image:
        state = self.get_state(time)
        assert state is not None
        slide_number = self.slide_timeline[state.name]
        return self.slide_images[slide_number]


class CharacterLayer(Layer):

    def __init__(
            self, timeline: pd.DataFrame,
            character_name: str, character_dir: str, character_column: str = 'character',
            status_column: str = 'status', initial_status: str = 'n',
            blink_per_minute: int = 3, blink_duration: float = 0.2) -> None:
        super().__init__(timeline)
        self.character_name = character_name
        self.character_imgs = {}
        self.eye_imgs = {}
        emotions = set(self.timeline[
            self.timeline[character_column] == character_name][status_column].unique())
        emotions.add(initial_status)
        for emotion in emotions:
            path = os.path.join(character_dir, f'{emotion}.png')
            self.character_imgs[emotion] = Image.open(path).convert('RGBA')
            eye_path = os.path.join(character_dir, f'{emotion}.eye.png')
            if os.path.exists(eye_path):
                eyes = [Image.open(eye_path).convert('RGBA')]
                for f in sorted(os.listdir(character_dir)):
                    x = f.split('.')
                    if f.startswith(f'{emotion}.eye.') and len(x) == 4:
                        eyes.append(Image.open(os.path.join(character_dir, f)).convert('RGBA'))
                self.eye_imgs[emotion] = eyes

        self.character_timeline = []
        status = initial_status
        for _, row in self.timeline.iterrows():
            if row[character_column] == character_name:
                status = row[status_column]
            self.character_timeline.append(status)

        self.blink_per_minute = blink_per_minute
        self.blink_duration = blink_duration

    def get_eye_state(self, time: float, state: pd.Series) -> int:
        emotion = self.character_timeline[state.name]
        if emotion not in self.eye_imgs:
            return -1
        elif len(self.eye_imgs[emotion]) == 1:
            return 0
        p_threshold = self.blink_per_minute * self.blink_duration / 60
        n = int(time / self.blink_duration)
        p = rand_from_string(f'{self.character_name}:{n}')
        if p < p_threshold:
            frame_duration = self.blink_duration / (len(self.eye_imgs[emotion]) - 1)
            t1 = time - n * self.blink_duration
            n1 = int(t1 / frame_duration)
            return min(n1 + 1, len(self.eye_imgs[emotion]) - 1)
        else:
            return 0

    def get_keys(self, time: float) -> tuple[Any, ...]:
        state = self.get_state(time)
        if state is None:
            return (None, None)
        emotion = self.character_timeline[state.name]
        eye = self.get_eye_state(time, state)
        return (emotion, eye)

    def render(self, time: float) -> Image.Image:
        state = self.get_state(time)
        assert state is not None
        emotion = self.character_timeline[state.name]
        base_img = self.character_imgs[emotion]
        if emotion in self.eye_imgs:
            eye_number = self.get_eye_state(time, state)
            eye_img = self.eye_imgs[emotion][eye_number]
            base_img = base_img.copy()
            base_img.alpha_composite(eye_img)
        return base_img


type_to_layer_cls = {
    'image': ImageLayer,
    'slide': SlideLayer,
    'character': CharacterLayer,
}


class LayerWithProperty(NamedTuple):

    name: str
    layer: Layer
    transform: Transform = Transform()
    offset: float = 0.


class Composition(Layer):

    def __init__(self, timeline: pd.DataFrame, size: tuple[int, int] = (1920, 1080)) -> None:
        super().__init__(timeline)
        self.layers: list[LayerWithProperty] = []
        self.name_to_layer: dict[str, LayerWithProperty] = {}
        self.animations: DefaultDict[str, list[Animation]] = defaultdict(list)
        self.size = size
        self.cache: LRUCache = LRUCache(maxsize=128)

    def init_from_config(self, layers_config: list[dict]) -> None:
        for cfg in layers_config:
            name = cfg.pop('name')
            kwargs = {'timeline': self.timeline}
            transform = Transform(
                anchor_point=normalize_2dvector(cfg.pop('anchor_point', 0.)),
                position=normalize_2dvector(cfg.pop('position', (self.size[0] / 2, self.size[1] / 2))),
                scale=normalize_2dvector(cfg.pop('scale', 1.)),
                opacity=cfg.pop('opacity', 1.),
            )
            layer_cls = type_to_layer_cls[cfg.pop('type')]
            kwargs.update(cfg)
            layer = layer_cls(**kwargs)
            self.add_layer(layer, name, transform)

        if 'animation' in self.timeline.columns:
            anim_frame = self.timeline[
                self.timeline['animation'].notnull() & (self.timeline['animation'] != '')]
            for _, row in anim_frame.iterrows():
                animations = parse_animation_command(
                    row['start_time'], row['end_time'], row['animation'])
                for layer_name, animation in animations:
                    self.add_animation(layer_name, animation)

    def get_keys(self, time: float) -> tuple[Any, ...]:
        layer_keys = []
        for layer_with_prop in self.layers:
            transform = layer_with_prop.transform
            layer = layer_with_prop.layer

            t = time - layer_with_prop.offset
            animations = self.animations[layer_with_prop.name]
            prop = self.animate_property(transform, animations, t)
            layer_keys.append(prop + layer.get_keys(t))
        return tuple(layer_keys)

    def add_layer(self, layer: Layer, name: str = '', transform: Transform = Transform()) -> None:
        if name == '':
            name = f'layer_{len(self.layers)}'
        if name in self.name_to_layer:
            raise KeyError(f'Layer with name {name} already exists')
        layer_with_prop = LayerWithProperty(name, layer, transform)
        self.name_to_layer[name] = layer_with_prop
        self.layers.append(layer_with_prop)

    def add_animation(self, name: str, animation: Animation) -> None:
        if name not in self.name_to_layer:
            raise KeyError(f'Layer with name {name} does not exist')
        self.animations[name].append(animation)

    def animate_property(
            self, transform: Transform, animations: Iterable[Animation], time: float) -> Transform:
        prop = transform
        for anim in animations:
            p = anim(time)
            anchor_point = (prop.anchor_point[0] + p.anchor_point[0], prop.anchor_point[1] + p.anchor_point[1])
            position = (prop.position[0] + p.position[0], prop.position[1] + p.position[1])
            scale = (prop.scale[0] * p.scale[0], prop.scale[1] * p.scale[1])
            opacity = prop.opacity * p.opacity
            prop = Transform(
                anchor_point=anchor_point, position=position, scale=scale, opacity=opacity)
        return prop

    def composite(self, base_img: Image.Image, layer: LayerWithProperty, animations: Iterable[Animation], time: float) -> Image.Image:
        t = time - layer.offset
        state = layer.layer.get_state(t)
        if state is None:
            return base_img
        component = layer.layer.render(t)
        w, h = component.size
        p = self.animate_property(layer.transform, animations, t)
        component = resize(component, p.scale)
        x = p.position[0] - p.scale[0] * w / 2 - p.anchor_point[0]
        y = p.position[1] - p.scale[1] * h / 2 - p.anchor_point[1]
        alpha_composite(
            base_img, component, position=(round(x), round(y)), opacity=p.opacity)
        return base_img

    def render(self, time: float) -> Image.Image:
        keys = self.get_keys(time)
        if keys in self.cache:
            return self.cache[keys]

        frame = Image.new('RGBA', self.size)
        for layer in self.layers:
            animations = self.animations[layer.name]
            self.composite(frame, layer, animations, time)
        self.cache[keys] = frame
        return frame

    def make_video(
            self, dst_path: str, start_time: float = 0.0, end_time: Optional[float] = None,
            codec: str = 'libx264', fps: float = 30.0) -> None:
        if end_time is None:
            end_time = self.duration
        times = np.arange(start_time, end_time, 1. / fps)
        writer = imageio.get_writer(
            dst_path, fps=fps, codec=codec, macro_block_size=None)
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self.render(t))
            writer.append_data(frame)
        writer.close()
        self.cache.clear()


def render_video(
        video_config: dict, timeline_path: str,
        audio_dir: str, dst_video_path: str) -> None:
    timeline = pd.read_csv(timeline_path)
    audio_df = get_voicevox_dataframe(audio_dir)
    timeline = pd.merge(timeline, audio_df, left_index=True, right_index=True)
    size = (video_config['width'], video_config['height'])
    scene = Composition(timeline=timeline, size=size)
    scene.init_from_config(video_config['layers'])
    scene.make_video(dst_video_path, fps=video_config['fps'])


def render_subtitle_video(
        video_path: str, subtitle_path: str, audio_path: str, dst_video_path: str) -> None:
    video_option_str = f"ass={subtitle_path}"
    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)
    output = ffmpeg.output(
        video_input.video, audio_input.audio, dst_video_path,
        vf=video_option_str, acodec='aac', ab='128k')
    output.run(overwrite_output=True)
