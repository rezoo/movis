import os
from typing import Optional, NamedTuple, Union, Any

from cachetools import LRUCache
import imageio
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from zunda.motion import Motion, MotionSequence
from zunda.utils import rand_from_string
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
        self.image: Optional[Image.Image] = None
        self.img_path = img_path

    def get_keys(self, time: float) -> tuple[int]:
        state = self.get_state(time)
        key = 1 if state is not None else 0
        return (key,)

    def render(self, time: float) -> Image.Image:
        if self.image is None:
            self.image = Image.open(self.img_path).convert('RGBA')
        return self.image


class SlideLayer(Layer):

    def __init__(
            self, timeline: pd.DataFrame,
            slide_path: str, slide_column: str = 'slide') -> None:
        super().__init__(timeline)
        self.slide_timeline = np.cumsum(self.timeline[slide_column])
        self.slide_path = slide_path
        self.slide_images: Optional[list[Image.Image]] = None

    def get_keys(self, time: float) -> tuple[Any, ...]:
        state = self.get_state(time)
        key = int(self.slide_timeline[state.name]) if state is not None else None
        return (key,)

    def render(self, time: float) -> Image.Image:
        state = self.get_state(time)
        assert state is not None
        slide_number = self.slide_timeline[state.name]
        if self.slide_images is None:
            slide_images = []
            for img in convert_from_path(self.slide_path):
                img = img.convert('RGBA')
                slide_images.append(img)
            self.slide_images = slide_images
        return self.slide_images[slide_number]


class CharacterLayer(Layer):

    def __init__(
            self, timeline: pd.DataFrame,
            character_name: str, character_dir: str, character_column: str = 'character',
            status_column: str = 'status', initial_status: str = 'n',
            blink_per_minute: int = 3, blink_duration: float = 0.2) -> None:
        super().__init__(timeline)
        self.character_name = character_name
        self.character_imgs: dict[str, Image.Image] = {}
        self.eye_imgs: dict[str, list[Image.Image]] = {}
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

        self.character_timeline: list[str] = []
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


class Attribute(NamedTuple):

    attr_name: str
    value_type: str


class Composition(Layer):

    def __init__(self, timeline: pd.DataFrame, size: tuple[int, int] = (1920, 1080)) -> None:
        super().__init__(timeline)
        self.layers: list[LayerWithProperty] = []
        self._name_to_layer: dict[str, LayerWithProperty] = {}
        self.motions: dict[tuple[str, str], Motion] = {}
        self.size = size
        self.cache: LRUCache = LRUCache(maxsize=128)

    def add_layers_from_config(self, layers_config: list[dict]) -> None:
        for cfg in layers_config:
            name = cfg.pop('name')
            kwargs = {'timeline': self.timeline}
            transform = Transform.create(
                anchor_point=cfg.pop('anchor_point', 0.),
                position=cfg.pop('position', (self.size[0] / 2, self.size[1] / 2)),
                scale=cfg.pop('scale', 1.),
                opacity=cfg.pop('opacity', 1.),
            )
            layer_cls = type_to_layer_cls[cfg.pop('type')]
            kwargs.update(cfg)
            layer = layer_cls(**kwargs)
            self.add_layer(layer, name, transform)

    def get_keys(self, time: float) -> tuple[Any, ...]:
        layer_keys = []
        for layer_with_prop in self.layers:
            layer = layer_with_prop.layer
            layer_time = time - layer_with_prop.offset
            p = self._get_current_transform(layer_with_prop, layer_time)
            layer_keys.append(p + layer.get_keys(layer_time))
        return tuple(layer_keys)

    def get_layer(self, name: str) -> LayerWithProperty:
        return self._name_to_layer[name]

    def add_layer(self, layer: Layer, name: str = '', transform: Transform = Transform()) -> None:
        if name == '':
            name = f'layer_{len(self.layers)}'
        if name in self.layers:
            raise KeyError(f'Layer with name {name} already exists')
        layer_with_prop = LayerWithProperty(name, layer, transform)
        self.layers.append(layer_with_prop)
        self._name_to_layer[name] = layer_with_prop

    @property
    def attributes(self) -> list[tuple[str, list[Attribute]]]:
        attrs: list[tuple[str, list[Attribute]]] = []
        for layer_with_prop in self.layers:
            name = layer_with_prop.name
            attrs.append((name, [
                Attribute('anchor_point', 'vector2d'),
                Attribute('position', 'vector2d'),
                Attribute('scale', 'vector2d'),
                Attribute('opacity', 'scalar'),
            ]))
        return attrs

    def get_current_attr(self, layer_name: str, attr_name: str, time: float = 0.) -> Union[float, tuple[float, float]]:
        layer_with_prop = self._name_to_layer[layer_name]
        if (layer_name, attr_name) in self.motions:
            motion = self.motions[(layer_name, attr_name)]
            return motion(time)
        else:
            value = getattr(layer_with_prop.transform, attr_name)
            return value

    def has_motion(self, layer_name: str, attr_name: str) -> bool:
        return (layer_name, attr_name) in self.motions

    def set_motion(self, layer_name: str, attr_name: str, motion: Motion) -> None:
        self.motions[(layer_name, attr_name)] = motion

    def get_motion(self, layer_name: str, attr_name: str, auto_create: bool = False) -> Motion:
        if self.has_motion(layer_name, attr_name):
            return self.motions[(layer_name, attr_name)]
        else:
            if auto_create:
                value = self.get_current_attr(layer_name, attr_name)
                motion = MotionSequence(default_value=value)
                self.set_motion(layer_name, attr_name, motion)
                return motion
            raise KeyError(f'No motion for {layer_name}.{attr_name}')

    def _get_current_transform(
            self, layer_with_prop: LayerWithProperty, layer_time: float) -> Transform:
        name = layer_with_prop.name
        return Transform(
            anchor_point=self.get_current_attr(name, 'anchor_point', layer_time),
            position=self.get_current_attr(name, 'position', layer_time),
            scale=self.get_current_attr(name, 'scale', layer_time),
            opacity=self.get_current_attr(name, 'opacity', layer_time))

    def composite(self, base_img: Image.Image, layer_with_prop: LayerWithProperty, time: float) -> Image.Image:
        layer_time = time - layer_with_prop.offset
        state = layer_with_prop.layer.get_state(layer_time)
        if state is None:
            return base_img
        component = layer_with_prop.layer.render(layer_time)
        w, h = component.size

        p = self._get_current_transform(layer_with_prop, layer_time)
        component = resize(component, p.scale)
        x = p.position[0] + (p.anchor_point[0] - w / 2) * p.scale[0]
        y = p.position[1] + (p.anchor_point[1] - h / 2) * p.scale[1]
        alpha_composite(
            base_img, component, position=(round(x), round(y)), opacity=p.opacity)
        return base_img

    def render(self, time: float) -> Image.Image:
        keys = self.get_keys(time)
        if keys in self.cache:
            return self.cache[keys]

        frame = Image.new('RGBA', self.size)
        for layer_with_prop in self.layers:
            self.composite(frame, layer_with_prop, time)
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
