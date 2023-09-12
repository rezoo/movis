from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio
import librosa
import numpy as np
from PIL import Image as PILImage

from .mixin import TimelineMixin
from .protocol import AUDIO_SAMPLING_RATE


class Image:
    """Still image layer to encapsulate various formats of image data and offer
    time-based keying.

    Args:
        img_file: the source of the image data. It can be a file path (``str`` or ``Path``),
            a `PIL.Image` object, or a two or three-dimensional ``numpy.ndarray`` with a shape of ``(H, W, C)``.
        duration: the duration for which the image should be displayed.
            Default is ``1000000.0`` (long enough time).
    """
    def __init__(
        self,
        img_file: str | Path | PILImage.Image | np.ndarray,
        duration: float = 1e6
    ) -> None:
        self.image: np.ndarray | None = None
        self._img_file: Path | None = None
        if isinstance(img_file, (str, Path)):
            self._img_file = Path(img_file)
            assert self._img_file.exists(), f"{self._img_file} does not exist"
        elif isinstance(img_file, PILImage.Image):
            image = np.asarray(img_file.convert("RGBA"))
            self.image = image
        elif isinstance(img_file, np.ndarray):
            if img_file.ndim == 2:
                assert img_file.dtype == np.uint8
                img = np.expand_dims(img_file, axis=-1)
                img = np.concatenate([
                    np.repeat(img, 3, axis=-1),
                    np.full_like(img, 255, dtype=np.uint8)],
                    axis=-1)
                self.image = img
            elif img_file.ndim == 3:
                assert img_file.shape[2] == 4, "Image must have 4 channels (RGBA)"
                self.image = img_file
            else:
                raise ValueError(f"Invalid img_file shape: {img_file.shape}")
        else:
            raise ValueError(f"Invalid img_file type: {type(img_file)}")

        self._duration = duration

    @property
    def duration(self) -> float:
        """The duration for which the image should be displayed."""
        return self._duration

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image with a tuple of `(width, height)`."""
        shape = self._read_image().shape[:2]
        return shape[1], shape[0]

    def get_key(self, time: float) -> bool:
        """Get the state index for the given time."""
        return 0 <= time < self.duration

    def _read_image(self) -> np.ndarray:
        if self.image is None:
            assert self._img_file is not None
            image = np.asarray(PILImage.open(self._img_file).convert("RGBA"))
            self.image = image
        return self.image

    def __call__(self, time: float) -> np.ndarray | None:
        return self._read_image()


class ImageSequence(TimelineMixin):
    """Image sequence layer to encapsulate various formats of images.

    Args:
        start_times:
            a sequence of start times for each image.
        end_times:
            a sequence of end times for each image.
        img_files:
            a sequence of image data. Each element can be a file path (``str`` or ``Path``),
            a `PIL.Image` object, or a two or four-dimensional ``numpy.ndarray`` with a shape of ``(H, W, C)``.
    """

    @classmethod
    def from_files(
        cls,
        img_files: Sequence[str | Path | PILImage.Image | np.ndarray],
        each_duration: float = 1.0
    ) -> "ImageSequence":
        """Create an ``ImageSequence`` object from a sequence of image files.

        Different from ``ImageSequence.__init__``, this method does not require the start and end times,
        and the duration for each image is set to ``each_duration``.

        Args:
            img_files:
                a sequence of image data. Each element can be a file path (``str`` or ``Path``),
                a `PIL.Image` object, or a two or four-dimensional ``numpy.ndarray`` with a shape of ``(H, W, C)``.
            each_duration:
                the duration for which each image should be displayed. Default is ``1.0``.

        Returns:
            An ``ImageSequence`` object.
        """
        start_times = np.arange(len(img_files)) * each_duration
        end_times = start_times + each_duration
        return cls(start_times.tolist(), end_times.tolist(), img_files)

    @classmethod
    def from_dir(
        cls,
        img_dir: str | Path,
        each_duration: float = 1.0
    ) -> "ImageSequence":
        """Create an ``ImageSequence`` object from a directory of image files.

        Args:
            img_dir:
                a directory containing image files.
            each_duration:
                the duration for which each image should be displayed. Default is ``1.0``.

        Returns:
            An ``ImageSequence`` object.
        """
        img_dir = Path(img_dir)
        exts = set([".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
        img_files = [
            p for p in sorted(img_dir.glob("*")) if p.is_file() and p.suffix in exts]
        return cls.from_files(img_files, each_duration)

    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        img_files: Sequence[str | Path | PILImage.Image | np.ndarray]
    ) -> None:
        super().__init__(start_times, end_times)
        self.img_files = img_files
        self.images: list[np.ndarray | None] = [None] * len(img_files)
        for i, img_file in enumerate(img_files):
            if isinstance(img_file, (str, Path)):
                img_file = Path(img_file)
                assert Path(img_file).exists(), f"{img_file} does not exist"
            elif isinstance(img_file, PILImage.Image):
                self.images[i] = np.asarray(img_file.convert("RGBA"))
            elif isinstance(img_file, np.ndarray):
                self.images[i] = img_file
            else:
                raise ValueError(f"Invalid img_file type: {type(img_file)}")

    def get_key(self, time: float) -> int:
        """Get the state index for the given time."""
        idx = self.get_state(time)
        if idx < 0:
            return -1
        return idx

    def __call__(self, time: float) -> np.ndarray | None:
        idx = self.get_state(time)
        if idx < 0:
            return None
        if self.images[idx] is None:
            img_file = self.img_files[idx]
            assert isinstance(img_file, (str, Path))
            image = np.asarray(PILImage.open(img_file).convert("RGBA"))
            self.images[idx] = image
        return self.images[idx]


class Video:
    """Video layer to encapsulate various formats of video data.

    Args:
        video_file:
            the source of the video data. It can be a file path (``str`` or ``Path``).
    """

    def __init__(self, video_file: str | Path) -> None:
        self.video_file = Path(video_file)
        self._reader = imageio.get_reader(video_file)
        meta_data = self._reader.get_meta_data()
        self._fps = meta_data["fps"]
        self._size = meta_data["size"]
        self._n_frame = meta_data["nframes"]
        self._duration = meta_data["duration"]

    @property
    def fps(self) -> float:
        """The frame rate of the video."""
        return self._fps

    @property
    def size(self) -> tuple[int, int]:
        """The size of the video with a tuple of ``(width, height)``."""
        return self._size

    @property
    def n_frame(self) -> int:
        """The number of frames in the video."""
        return self._n_frame

    @property
    def duration(self) -> float:
        """The duration of the video."""
        return self._duration

    def get_key(self, time: float) -> int:
        """Get the state index for the given time."""
        if time < 0 or self.duration < time:
            return -1
        frame_index = int(time * self._fps)
        return frame_index

    def __call__(self, time: float) -> np.ndarray | None:
        frame_index = int(time * self._fps)
        frame = self._reader.get_data(frame_index)
        pil_frame = PILImage.fromarray(frame).convert("RGBA")
        return np.asarray(pil_frame)


class Audio:
    """Audio layer to encapsulate various formats of audio data.

    Args:
        audio_file:
            the source of the audio data. It can be a file path (``str`` or ``Path``).
    """

    def __init__(self, audio_file: str | Path, audio_level: float = 0.0) -> None:
        self._audio_file: Path | None = None
        self._audio: np.ndarray | None = None
        self._duration: float | None = None
        if isinstance(audio_file, (str, Path)):
            self._audio_file = Path(audio_file)
            assert self._audio_file.exists(), f"{self._audio_file} does not exist"
        else:
            raise ValueError(f"Invalid audio_file type: {type(audio_file)}")

    def _load_audio(self) -> np.ndarray:
        if self._audio is None:
            audio, _ = librosa.load(self._audio_file, sr=AUDIO_SAMPLING_RATE, mono=False)
            if audio.ndim == 1:
                audio = np.broadcast_to(audio[None, :], (2, len(audio)))
            self._audio = audio
        return self._audio

    @property
    def audio_file(self) -> Path | None:
        """The file path of the audio data."""
        return self._audio_file

    @property
    def duration(self) -> float:
        """The duration of the audio data."""
        if self._duration is not None:
            return self._duration
        duration = librosa.get_duration(path=self._audio_file)
        self._duration = duration
        return duration

    def __call__(self, time: float) -> np.ndarray | None:
        return None

    def get_key(self, time: float) -> int:
        """Get the state index for the given time.

        Note:
            This method always returns a constant value because the audio data does not affect the image data.
        """
        return 0

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        """Get the audio data for the given time range.

        Args:
            start_time: the start time of the audio data.
            end_time: the end time of the audio data.

        Returns:
            The audio data for the given time range. If no audio data is found, ``None`` is returned.
        """
        audio = self._load_audio()
        start_index = int(start_time * AUDIO_SAMPLING_RATE)
        end_index = int(end_time * AUDIO_SAMPLING_RATE)
        return audio[:, start_index:end_index]


class AudioSequence:
    """Audio sequence layer to handle multiple audio files.

    Args:
        start_times:
            a sequence of start times for each audio.
        end_times:
            a sequence of end times for each audio.
        audio_files:
            a sequence of audio data. Each element can be a file path (``str`` or ``Path``).
        """

    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        audio_files: Sequence[str | Path],
    ) -> None:
        assert len(start_times) == len(end_times) == len(audio_files)
        self.start_times = np.asarray(start_times, dtype=float)
        self.end_times = np.asarray(end_times, dtype=float)
        self.audio_files = list(audio_files)
        self._audio: list[np.ndarray | None] = [None] * len(audio_files)

    @property
    def duration(self) -> float:
        """The duration of the audio sequence."""
        return self.end_times[-1]

    def __call__(self, time: float) -> np.ndarray | None:
        return None

    def get_key(self, time: float) -> int:
        """Get the state index for the given time.

        Note:
            This method always returns a constant value because the audio data does not affect the image data."""
        return 0

    def _load_audio(self, index: int) -> np.ndarray:
        a = self._audio[index]
        if a is None:
            audio_file = self.audio_files[index]
            if isinstance(audio_file, (str, Path)):
                a_i, _ = librosa.load(audio_file, sr=AUDIO_SAMPLING_RATE, mono=False)
                if a_i.ndim == 1:
                    a_i = np.broadcast_to(a_i[None, :], (2, len(a_i)))
            else:
                raise ValueError(f"Invalid audio_file type: {type(audio_file)}")
            self._audio[index] = a_i
            return a_i
        return a

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        """Get the audio data for the given time range.

        Args:
            start_time:
                the start time of the audio data.
            end_time:
                the end time of the audio data.

        Returns:
            The audio data for the given time range. If no audio data is found, ``None`` is returned.
        """
        assert start_time < end_time
        audio: list[np.ndarray] = []
        for i, (s, e) in enumerate(zip(self.start_times, self.end_times)):
            if s < end_time and e > start_time:
                audio_i = self._load_audio(i)
                start_index = int(max(start_time - s, 0) * AUDIO_SAMPLING_RATE)
                end_index = int(min(end_time - s, e - s) * AUDIO_SAMPLING_RATE)
                if len(audio_i) < end_index:
                    # Pad with zeros
                    audio_i = np.pad(audio_i, ((0, 0), (0, end_index - len(audio_i))))
                audio.append(audio_i[:, start_index:end_index])
        if len(audio) == 0:
            return None
        return np.concatenate(audio, axis=1)
