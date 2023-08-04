from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (Any, Hashable, NamedTuple, Optional, Protocol, Sequence,
                    Union)

import imageio
import numpy as np
from cachetools import LRUCache
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from zunda.motion import Motion
from zunda.transform import Transform, alpha_composite, alpha_composite_numpy, resize
from zunda.utils import normalize_2dvector, rand_from_string


class Layer(Protocol):
    @property
    def duration(self):
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        raise NotImplementedError

    def __call__(self, time: float) -> Optional[np.ndarray]:
        raise NotImplementedError


class TimelineMixin:
    def __init__(
        self, start_times: Sequence[float], end_times: Sequence[float]
    ) -> None:
        assert len(start_times) == len(
            end_times
        ), f"{len(start_times)} != {len(end_times)}"
        self.start_times: np.ndarray = np.asarray(start_times, dtype=float)
        self.end_times: np.ndarray = np.asarray(end_times, dtype=float)

    def get_state(self, time: float) -> int:
        idx = self.start_times.searchsorted(time, side="right") - 1
        if idx >= 0 and self.end_times[idx] > time:
            return int(idx)
        else:
            return -1

    @property
    def duration(self):
        return self.end_times[-1] - self.start_times[0]


class ImageLayer:
    def __init__(
        self, duration: float, img_file: Union[str, Path, Image.Image, np.ndarray]
    ) -> None:
        self.image: Optional[np.ndarray] = None
        self._img_file: Optional[Path] = None
        if isinstance(img_file, (str, Path)):
            self._img_file = Path(img_file)
            assert self._img_file.exists(), f"{self._img_file} does not exist"
        elif isinstance(img_file, Image.Image):
            image = np.asarray(img_file.convert("RGBA"))
            self.image = image
        elif isinstance(img_file, np.ndarray):
            self.image = img_file
        else:
            raise ValueError(f"Invalid img_file type: {type(img_file)}")

        self._duration = duration

    @property
    def duration(self):
        return self._duration

    def get_key(self, time: float) -> bool:
        return 0 <= time < self.duration

    def __call__(self, time: float) -> Optional[np.ndarray]:
        if self.image is None:
            image = np.asarray(Image.open(self._img_file).convert("RGBA"))
            self.image = image
        return self.image


class VideoLayer:
    def __init__(self, video_file: Union[str, Path]) -> None:
        self.video_file = Path(video_file)
        self.reader = imageio.get_reader(video_file)
        meta_data = self.reader.get_meta_data()
        self.fps = meta_data["fps"]
        self.n_frames = meta_data["nframes"]
        self._duration = meta_data["duration"]

    @property
    def duration(self):
        return self._duration

    def get_key(self, time: float) -> int:
        if time < 0 or self.duration < time:
            return -1
        frame_index = int(time * self.fps)
        return frame_index

    def __call__(self, time: float) -> Optional[np.ndarray]:
        frame_index = int(time * self.fps)
        frame = self.reader.get_data(frame_index)
        frame = Image.fromarray(frame).convert("RGBA")
        return np.asarray(frame)


class SlideLayer(TimelineMixin):
    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        slide_file: Union[str, Path],
        slide_counter: Sequence[int],
    ) -> None:
        super().__init__(start_times, end_times)
        self.slide_timeline = np.cumsum(slide_counter)
        self.slide_file = slide_file
        self.slide_images: Optional[list[np.ndarray]] = None

    def get_key(self, time: float) -> int:
        idx = self.get_state(time)
        key = int(self.slide_timeline[idx]) if 0 < idx else -1
        return key

    def __call__(self, time: float) -> Optional[np.ndarray]:
        idx = self.get_state(time)
        if idx < 0:
            return None
        slide_number = self.slide_timeline[idx]
        if self.slide_images is None:
            slide_images = []
            for img in convert_from_path(self.slide_file):
                img = np.asarray(img.convert("RGBA"))
                slide_images.append(img)
            self.slide_images = slide_images
        return self.slide_images[slide_number]


class CharacterLayer(TimelineMixin):
    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        character_name: str,
        character_dir: Union[str, Path],
        characters: Sequence[str],
        character_status: Sequence[str],
        initial_status: str = "n",
        blink_per_minute: int = 3,
        blink_duration: float = 0.2,
    ) -> None:
        assert len(start_times) == len(characters) == len(character_status)
        super().__init__(start_times, end_times)
        self.character_name = character_name
        self.character_imgs: dict[str, Union[Path, np.ndarray]] = {}
        self.eye_imgs: dict[str, list[Union[Path, np.ndarray]]] = {}
        character_dir = Path(character_dir)
        emotions = set()
        for character, status in zip(characters, character_status):
            if character == character_name:
                emotions.add(status)
        emotions.add(initial_status)
        for emotion in emotions:
            path = Path(character_dir) / f"{emotion}.png"
            self.character_imgs[emotion] = path
            eye_path = Path(character_dir) / f"{emotion}.eye.png"
            if eye_path.exists():
                eyes: list[Union[Path, np.ndarray]] = [eye_path]
                for f in character_dir.iterdir():
                    x = f.name.split(".")
                    if f.name.startswith(f"{emotion}.eye.") and len(x) == 4:
                        eyes.append(f)
                self.eye_imgs[emotion] = eyes

        self.character_timeline: list[str] = []
        status = initial_status
        for current_character, current_status in zip(characters, character_status):
            if current_character == character_name:
                status = current_status
            self.character_timeline.append(status)

        self.blink_per_minute = blink_per_minute
        self.blink_duration = blink_duration

    def get_eye_state(self, time: float, idx: int) -> int:
        emotion = self.character_timeline[idx]
        if emotion not in self.eye_imgs:
            return -1
        elif len(self.eye_imgs[emotion]) == 1:
            return 0
        p_threshold = self.blink_per_minute * self.blink_duration / 60
        n = int(time / self.blink_duration)
        p = rand_from_string(f"{self.character_name}:{n}")
        if p < p_threshold:
            frame_duration = self.blink_duration / (len(self.eye_imgs[emotion]) - 1)
            t1 = time - n * self.blink_duration
            n1 = int(t1 / frame_duration)
            return min(n1 + 1, len(self.eye_imgs[emotion]) - 1)
        else:
            return 0

    def get_key(self, time: float) -> tuple[str, int]:
        idx = self.get_state(time)
        if idx < 0:
            return ('', -1)
        emotion = self.character_timeline[idx]
        eye = self.get_eye_state(time, idx)
        return (emotion, eye)

    def __call__(self, time: float) -> Optional[np.ndarray]:
        idx = self.get_state(time)
        if idx < 0:
            return None
        emotion = self.character_timeline[idx]
        character = self.character_imgs[emotion]
        if isinstance(character, Path):
            base_img = np.asarray(Image.open(character).convert("RGBA"))
            self.character_imgs[emotion] = base_img
        else:
            base_img = character

        if emotion in self.eye_imgs:
            eye_number = self.get_eye_state(time, idx)
            eye = self.eye_imgs[emotion][eye_number]
            if isinstance(eye, Path):
                eye_img = np.asarray(Image.open(eye).convert("RGBA"))
                self.eye_imgs[emotion][eye_number] = eye_img
            else:
                eye_img = eye
            base_img = alpha_composite(base_img, eye_img)
        return base_img


class AttributeType(Enum):
    SCALAR = 0
    VECTOR2D = 1
    VECTOR3D = 2
    ANGLE = 3
    COLOR = 4

    @staticmethod
    def from_string(s: str) -> "AttributeType":
        if s == 'scalar':
            return AttributeType.SCALAR
        elif s == 'vector2d':
            return AttributeType.VECTOR2D
        elif s == 'vector3d':
            return AttributeType.VECTOR3D
        elif s == 'angle':
            return AttributeType.ANGLE
        else:
            raise ValueError(f"Unknown attribute type: {s}")


class Attribute(NamedTuple):

    attr_name: str
    value_type: AttributeType


class CacheType(Enum):
    COMPOSITION = 0
    LAYER = 1


@dataclass(eq=False)
class LayerProperty:

    name: str
    layer: Layer
    transform: Transform = Transform()
    offset: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self) -> None:
        self.end_time = self.end_time if self.end_time == 0.0 else self.layer.duration
        self._motions: dict[str, Motion] = {}

    @property
    def attributes(self) -> tuple[Attribute, ...]:
        return (
            Attribute("anchor_point", AttributeType.VECTOR2D),
            Attribute("position", AttributeType.VECTOR2D),
            Attribute("scale", AttributeType.VECTOR2D),
            Attribute("opacity", AttributeType.SCALAR),
        )

    def __call__(
        self, attr_name: str, layer_time: float = 0.0
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if attr_name in self._motions:
            motion = self._motions[attr_name]
            return motion(layer_time)
        else:
            value = getattr(self.transform, attr_name)
            return value

    def has_motion(self, attr_name: str) -> bool:
        return attr_name in self._motions

    def set_motion(self, attr_name: str, motion: Motion) -> Motion:
        self._motions[attr_name] = motion
        return motion

    def enable_motion(self, attr_name: str) -> Motion:
        if self.has_motion(attr_name):
            return self._motions[attr_name]
        else:
            value = self(attr_name)
            return self.set_motion(attr_name, Motion(default_value=value))

    def disable_motion(self, attr_name: str) -> None:
        if self.has_motion(attr_name):
            del self._motions[attr_name]

    def get_current_transform(self, layer_time: float) -> Transform:
        opacity = self("opacity", layer_time)
        return Transform(
            anchor_point=normalize_2dvector(self("anchor_point", layer_time)),
            position=normalize_2dvector(self("position", layer_time)),
            scale=normalize_2dvector(self("scale", layer_time)),
            opacity=float(opacity),
        )


class Composition:
    def __init__(
        self, size: tuple[int, int] = (1920, 1080), duration: float = 1.0
    ) -> None:
        self.layers: list[LayerProperty] = []
        self._name_to_layer: dict[str, LayerProperty] = {}
        self.size = size
        self._duration = duration
        self.cache: LRUCache = LRUCache(maxsize=128)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def __getitem__(self, key: str) -> LayerProperty:
        return self._name_to_layer[key]

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        layer_keys: list[Hashable] = [CacheType.COMPOSITION]
        for layer_prop in self.layers:
            layer = layer_prop.layer
            layer_time = time - layer_prop.offset
            if layer_time < layer_prop.start_time or layer_prop.end_time <= layer_time:
                layer_keys.append(f"__{layer_prop.name}__")
            else:
                p = layer_prop.get_current_transform(layer_time)
                layer_keys.append((p, layer.get_key(layer_time)))
        return tuple(layer_keys)

    def add_layer(
        self,
        layer: Layer,
        name: Optional[str] = None,
        transform: Optional[Transform] = None,
        offset: float = 0.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> LayerProperty:
        if name is None:
            name = f"layer_{len(self.layers)}"
        if name in self.layers:
            raise KeyError(f"Layer with name {name} already exists")
        end_time = end_time if end_time is not None else layer.duration
        transform = transform if transform is not None else Transform()
        layer_prop = LayerProperty(
            name,
            layer,
            transform,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
        )
        self.layers.append(layer_prop)
        self._name_to_layer[name] = layer_prop
        return layer_prop

    def _get_or_resize(
        self, layer_prop: LayerProperty, layer_time: float,
        component: np.ndarray, scale: tuple[float, float]
    ) -> np.ndarray:
        layer_state = layer_prop.layer.get_key(layer_time)
        key = (CacheType.LAYER, layer_prop.name, layer_state, scale)
        if key in self.cache:
            return self.cache[key]
        img = resize(component, scale)
        self.cache[key] = img
        return img

    def composite(
        self, base_img: np.ndarray, layer_prop: LayerProperty, time: float
    ) -> np.ndarray:
        layer_time = time - layer_prop.offset
        if layer_time < layer_prop.start_time or layer_prop.end_time <= layer_time:
            return base_img
        component = layer_prop.layer(layer_time)
        if component is None:
            return base_img
        h, w = component.shape[:2]

        p = layer_prop.get_current_transform(layer_time)
        component = self._get_or_resize(layer_prop, layer_time, component, p.scale)
        x = p.position[0] + (p.anchor_point[0] - w / 2) * p.scale[0]
        y = p.position[1] + (p.anchor_point[1] - h / 2) * p.scale[1]
        base_img = alpha_composite_numpy(
            base_img, component, position=(round(x), round(y)), opacity=p.opacity)
        return base_img

    def __call__(self, time: float) -> Optional[np.ndarray]:
        key = self.get_key(time)
        if key in self.cache:
            return self.cache[key]

        frame = np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        for layer_prop in self.layers:
            frame = self.composite(frame, layer_prop, time)
        self.cache[key] = frame
        return frame

    def make_video(
        self,
        dst_file: Union[str, Path],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        codec: str = "libx264",
        fps: float = 30.0,
    ) -> None:
        if end_time is None:
            end_time = self.duration
        times = np.arange(start_time, end_time, 1.0 / fps)
        writer = imageio.get_writer(
            dst_file, fps=fps, codec=codec, macro_block_size=None
        )
        for t in tqdm(times, total=len(times)):
            frame = np.asarray(self(t))
            writer.append_data(frame)
        writer.close()
        self.cache.clear()
