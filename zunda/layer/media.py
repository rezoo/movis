from pathlib import Path
from typing import Optional, Sequence, Union

import imageio
import numpy as np
from PIL import Image as PILImage

from zunda.imgproc import alpha_composite
from zunda.layer.mixin import TimelineMixin
from zunda.util import rand_from_string

try:
    from pdf2image import convert_from_path
    pdf2image_available = True
except ImportError:
    pdf2image_available = False


class Image:
    def __init__(
        self, img_file: Union[str, Path, PILImage.Image, np.ndarray], duration: float = 1.0
    ) -> None:
        self.image: Optional[np.ndarray] = None
        self._img_file: Optional[Path] = None
        if isinstance(img_file, (str, Path)):
            self._img_file = Path(img_file)
            assert self._img_file.exists(), f"{self._img_file} does not exist"
        elif isinstance(img_file, PILImage.Image):
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
            image = np.asarray(PILImage.open(self._img_file).convert("RGBA"))
            self.image = image
        return self.image


class ImageSequence(TimelineMixin):

    @classmethod
    def from_files(cls, img_files: Sequence[Union[str, Path]], each_duration: float = 1.0) -> "ImageSequence":
        start_times = np.arange(len(img_files)) * each_duration
        end_times = start_times + each_duration
        return cls(start_times.tolist(), end_times.tolist(), img_files)

    @classmethod
    def from_dir(cls, img_dir: Union[str, Path], each_duration: float = 1.0) -> "ImageSequence":
        img_dir = Path(img_dir)
        exts = set([".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
        img_files = [
            p for p in sorted(img_dir.glob("*")) if p.is_file() and p.suffix in exts]
        return cls.from_files(img_files, each_duration)

    def __init__(
        self, start_times: Sequence[float], end_times: Sequence[float], img_files: Sequence[Union[str, Path, np.ndarray]]
    ) -> None:
        super().__init__(start_times, end_times)
        self.img_files = img_files
        self.images: list[Optional[np.ndarray]] = [None] * len(img_files)
        for i, img_file in enumerate(img_files):
            if isinstance(img_file, (str, Path)):
                img_file = Path(img_file)
                assert Path(img_file).exists(), f"{img_file} does not exist"
            elif isinstance(img_file, np.ndarray):
                self.images[i] = img_file
            else:
                raise ValueError(f"Invalid img_file type: {type(img_file)}")

    def get_key(self, time: float) -> int:
        idx = self.get_state(time)
        if idx < 0:
            return -1
        return idx

    def __call__(self, time: float) -> Optional[np.ndarray]:
        idx = self.get_state(time)
        if idx < 0:
            return None
        if self.images[idx] is None:
            image = np.asarray(PILImage.open(self.img_files[idx]).convert("RGBA"))
            self.images[idx] = image
        return self.images[idx]


class Video:
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
        frame = PILImage.fromarray(frame).convert("RGBA")
        return np.asarray(frame)


class Slide(TimelineMixin):
    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        slide_file: Union[str, Path],
        slide_counter: Sequence[int],
    ) -> None:
        if not pdf2image_available:
            raise ImportError("pdf2image is not installed")
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


class Character(TimelineMixin):
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
            base_img = np.asarray(PILImage.open(character).convert("RGBA"))
            self.character_imgs[emotion] = base_img
        else:
            base_img = character

        if emotion in self.eye_imgs:
            eye_number = self.get_eye_state(time, idx)
            eye = self.eye_imgs[emotion][eye_number]
            if isinstance(eye, Path):
                eye_img = np.asarray(PILImage.open(eye).convert("RGBA"))
                self.eye_imgs[emotion][eye_number] = eye_img
            else:
                eye_img = eye
            base_img = alpha_composite(base_img, eye_img)
        return base_img
