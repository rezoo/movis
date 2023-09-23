from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image as PILImage

from ..imgproc import alpha_composite
from ..layer.mixin import TimelineMixin

try:
    from pdf2image import convert_from_path
    pdf2image_available = True
except ImportError:
    pdf2image_available = False


class Slide(TimelineMixin):
    """Slide layer for showing PDF slides.

    Many presentation videos use slides to explain the contents.
    This layer directly show the slides in the specified PDF file for convenience.

    .. note::
        This layer requires `pdf2image <https://github.com/Belval/pdf2image>`_ to be installed.

    Args:
        start_times:
            The start times for displaying slides.
        end_times:
            The end times for displaying slides.
        slide_file:
            The path to the PDF file containing the slides.
        slide_counter:
            The slide number for each slide. If ``None``, the slide number is automatically assigned.

    Examples:
        >>> from movis.contrib.commentary import Slide
        >>> # Show 0 page from 0.0 to 5.0 seconds, and 1 page from 5.0 to 10.0 seconds.
        >>> slide1 = Slide(start_times=[0.0, 5.0], end_times=[5.0, 10.0], slide_file="slides.pdf")
        >>> # Show 0 page from 0.0 to 5.0 seconds, and 1 page from 5.0 to 15.0 seconds.
        >>> slide2 = Slide([0., 5., 10.], [5., 10., 15.], "slides.pdf", slide_counter=[0, 1, 1])
    """
    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        slide_file: str | PathLike,
        slide_counter: Sequence[int] | None = None,
    ) -> None:
        if not pdf2image_available:
            raise ImportError("pdf2image is not installed")
        super().__init__(start_times, end_times)
        if slide_counter is None:
            self.slide_timeline = np.arange(len(start_times))
        else:
            self.slide_timeline = np.asarray(slide_counter)
        self.slide_file = slide_file
        self.slide_images: list[np.ndarray] | None = None

    def get_key(self, time: float) -> int:
        """Return the key for caching."""
        idx = self.get_state(time)
        key = int(self.slide_timeline[idx]) if 0 < idx else -1
        return key

    def __call__(self, time: float) -> np.ndarray | None:
        idx = self.get_state(time)
        if idx < 0:
            return None
        slide_number = self.slide_timeline[idx]
        if self.slide_images is None:
            slide_images = []
            for img in convert_from_path(Path(self.slide_file)):
                img_np = np.asarray(img.convert("RGBA"))
                slide_images.append(img_np)
            self.slide_images = slide_images
        return self.slide_images[slide_number]


class Character(TimelineMixin):
    """Character layer for showing and animating characters.

    Some presentation videos may place characters who speak with different facial expressions
    depending on the situation. This layer mimics such a situation.

    Args:
        start_times:
            A list of start times in seconds at which the character's status will change.
            Should have the same length as `characters` and `character_status`.
        end_times:
            A list of end times in seconds corresponding to each status duration.
        character_name:
            The name of the character that this layer will manage.
        character_dir:
            The directory where images corresponding to the character's emotions are stored.
            For each emotion, there should be an image named `<emotion>.png`
            and an optional eye image named `<emotion>.eye.png`, `<emotion>.eye.0.png`, `<emotion>.eye.1.png`, ...
        characters:
            A list of character names for which the timeline events are defined.
            Should have the same length as ``start_times`` and ``character_status``.
        character_status:
            A list specifying the emotion or state of the character at the corresponding start time.
            Should have the same length as `start_times` and `characters`.
        initial_status:
            The initial emotion or state of the character before any timeline events occur.
            Defaults to "n" (Normal).
        blink_per_minute (int, optional):
            Number of times the character blinks per minute. Defaults to 3.
        blink_duration (float, optional):
            Duration of a single blink in seconds. Defaults to 0.2.

    Examples:
        >>> from movis.contrib.commentary import Character
        >>> # Show a character named "alice" from 0.0 to 10.0 seconds.
        >>> # Alice's emotion does not change and only displays "character/alice/n.png".
        >>> alice = Character([0.0, 5.0], [5.0, 10.0], "alice", "character/alice", ["alice", "bob"], ["n", "h"])
        >>> # Show a character named "bob" from 0.0 to 10.0 seconds.
        >>> # Initially Bob's emotion is "n" (Normal), but it changes to "h" (Happy) at 5.0 seconds.
        >>> # In this case, "n" means "character/bob/n.png" and "h" means "character/bob/h.png".
        >>> bob = Character([0.0, 5.0], [5.0, 10.0], "bob", "character/bob", ["alice", "bob"], ["n", "h"])
        >>> # If "character/bob" contains "n.eye.png" and "n.eye.0.png, "n.eye.1.png", ..., "n.eye.N.png",
        >>> # in the "n" state, Bob's eyes blink an average of three times per minute. and duration is 0.2 seconds.
    """
    def __init__(
        self,
        start_times: Sequence[float],
        end_times: Sequence[float],
        character_name: str,
        character_dir: str | PathLike,
        characters: Sequence[str],
        character_status: Sequence[str],
        initial_status: str = "n",
        blink_per_minute: int = 3,
        blink_duration: float = 0.2,
    ) -> None:
        assert len(start_times) == len(characters) == len(character_status)
        super().__init__(start_times, end_times)
        self.character_name = character_name
        self.character_imgs: dict[str, Path | np.ndarray] = {}
        self.eye_imgs: dict[str, list[Path | np.ndarray]] = {}
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
                eyes: list[Path | np.ndarray] = [eye_path]
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

    def _get_eye_state(self, time: float, idx: int) -> int:

        def rand_from_string(string: str, seed: int = 0) -> float:
            string = f"{seed}:{string}"
            s = hashlib.sha224(f"{seed}:{string}".encode("utf-8")).digest()
            x = np.frombuffer(s, dtype=np.uint32)[0]
            return np.random.RandomState(x).rand()

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
        """Return the key for caching."""
        idx = self.get_state(time)
        if idx < 0:
            return ('', -1)
        emotion = self.character_timeline[idx]
        eye = self._get_eye_state(time, idx)
        return (emotion, eye)

    def __call__(self, time: float) -> np.ndarray | None:
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
            eye_number = self._get_eye_state(time, idx)
            eye = self.eye_imgs[emotion][eye_number]
            if isinstance(eye, Path):
                eye_img = np.asarray(PILImage.open(eye).convert("RGBA"))
                self.eye_imgs[emotion][eye_number] = eye_img
            else:
                eye_img = eye
            base_img = alpha_composite(base_img, eye_img)
        return base_img
