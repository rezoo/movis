import os
from typing import List

from cachetools import LRUCache
import ffmpeg
import numpy as np
import pandas as pd
import imageio
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from zunda.utils import _get_audio_dataframe


class Layer(object):

    def __init__(
            self, name: str, timeline: pd.DataFrame, scale: float = 1.0, position: tuple[int, int] = (0, 0)):
        self.name = name
        self.timeline = timeline
        self.position = position
        self.scale = scale

    def get_keys(self, time: float):
        return (self.position, self.scale)

    def get_state(self, time: float):
        subset = self.timeline[(self.timeline['start_time'] <= time) & (time < self.timeline['end_time'])]
        if subset.empty:
            return None
        return subset.iloc[0]

    def render(self, time: float, state: pd.Series) -> Image.Image:
        raise NotImplementedError


class ImageLayer(Layer):

    def __init__(self, img_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = Image.open(img_path).convert('RGBA')

    def render(self, time: float, state: pd.Series) -> Image.Image:
        return self.image


class SlideLayer(Layer):

    def __init__(self, slide_path: str, slide_column: str = 'slide', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slide_timeline = np.cumsum(self.timeline[slide_column])
        self.slide_images = self._get_slide_imgs(slide_path)

    def _get_slide_imgs(self, slide_path: str):
        slide_images = []
        for img in convert_from_path(slide_path):
            img = img.convert('RGBA')
            slide_images.append(img)
        return slide_images

    def get_keys(self, time: float):
        state = self.get_state(time)
        return super().get_keys(time) + (state.name,)

    def render(self, time: float, state: pd.Series) -> Image.Image:
        slide_number = self.slide_timeline[state.name]
        return self.slide_images[slide_number]


class CharacterLayer(Layer):

    def __init__(
            self, character_dir: str, character_column: str = 'character',
            status_column: str = 'status', initial_status='n', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.character_imgs = {}
        emotions = set(self.timeline[self.timeline[character_column] == self.name][status_column].unique())
        emotions.add(initial_status)
        for emotion in emotions:
            path = os.path.join(character_dir, f'{emotion}.png')
            self.character_imgs[emotion] = Image.open(path).convert('RGBA')

        self.character_timeline = []
        status = initial_status
        for _, row in self.timeline.iterrows():
            if row[character_column] == self.name:
                status = row[status_column]
            self.character_timeline.append(status)

    def get_keys(self, time: float):
        state = self.get_state(time)
        emotion = self.character_timeline[state.name]
        return super().get_keys(time) + (emotion,)

    def render(self, time: float, state: pd.Series) -> Image.Image:
        emotion = self.character_timeline[state.name]
        return self.character_imgs[emotion]


type_to_class = {
    'image': ImageLayer,
    'slide': SlideLayer,
    'character': CharacterLayer,
}


class Scene(object):

    def __init__(
            self, layers_config: List[dict], timeline: pd.DataFrame, size: tuple[int, int] = (1920, 1080)):
        self.layers = []
        self.size = size
        self.timeline = timeline

        self.cache = LRUCache(maxsize=256)

        for layer in layers_config:
            kwargs = {
                'timeline': self.timeline,
                'name': layer.pop('name'),
                'position': tuple(layer.pop('position')),
                'scale': layer.pop('scale'),
            }
            cls = type_to_class[layer.pop('type')]
            kwargs.update(layer)
            self.add_layer(cls(**kwargs))

    def add_layer(self, layer: any):
        self.layers.append(layer)

    def resize(self, img: Image, scale: float = 1.0) -> Image.Image:
        if scale == 1.0:
            return img
        w, h = img.size
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

    def get_frame(self, time: float) -> Image.Image:
        keys = tuple([layer.get_keys(time) for layer in self.layers])
        if keys in self.cache:
            return self.cache[keys]

        frame = Image.new('RGBA', self.size)
        for layer in self.layers:
            state = layer.get_state(time)
            if state is None:
                continue
            component = layer.render(time, state)
            component = self.resize(component, layer.scale)
            frame.alpha_composite(component, layer.position)
        self.cache[keys] = frame
        return frame

    def render(self, dst_path: str, codec: str = 'libx264', fps: float = 30.0) -> None:
        length = self.timeline['end_time'].max()
        times = np.arange(0, length, 1. / fps)
        writer = imageio.get_writer(
            dst_path, fps=fps, codec=codec, macro_block_size=None)
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self.get_frame(t))
            writer.append_data(frame)
        writer.close()


def render_video(
        video_config: dict, timeline_path: str,
        audio_dir: str, dst_video_path: str, fps: float = 30.0) -> None:
    timeline = pd.read_csv(timeline_path)
    audio_df = _get_audio_dataframe(audio_dir)
    timeline = pd.merge(timeline, audio_df, left_index=True, right_index=True)
    size = (video_config['width'], video_config['height'])
    scene = Scene(video_config['layers'], timeline, size=size)
    scene.render(dst_video_path, fps=video_config['fps'])


def render_subtitle_video(
        video_path: str, subtitle_path: str, audio_path: str, dst_video_path: str) -> None:
    video_option_str = f"ass={subtitle_path}"
    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)
    output = ffmpeg.output(
        video_input.video, audio_input.audio, dst_video_path,
        vf=video_option_str, acodec='aac', ab='128k')
    output.run(overwrite_output=True)
