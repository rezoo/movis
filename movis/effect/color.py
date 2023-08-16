import cv2
import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType


class FillColor(AttributesMixin):
    def __init__(self, color: tuple[int, int, int] = (255, 255, 255)):
        self.color = Attribute(color, AttributeType.COLOR)

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        rgb_image = np.full_like(
            prev_image[:, :, :3], np.round(self.color(time)).astype(np.uint8))
        alpha_image = prev_image[:, :, 3:]
        return np.concatenate([rgb_image, alpha_image], axis=2)


class HSLShift(AttributesMixin):
    def __init__(self, hue: float = 0.0, saturation: float = 0.0, luminance: float = 0.0):
        self.hue = Attribute(hue, AttributeType.SCALAR)
        self.saturation = Attribute(saturation, AttributeType.SCALAR)
        self.luminance = Attribute(luminance, AttributeType.SCALAR)

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        hsv_image = cv2.cvtColor(prev_image[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
        dh = self.hue(time).astype(np.float32)
        ds = self.saturation(time).astype(np.float32)
        dl = self.luminance(time).astype(np.float32)
        h = np.round((h + dh * (127.5 / 180)) % 255).astype(np.uint8)
        s = np.clip(np.round(s + ds * 255), 0, 255).astype(np.uint8)
        v = np.clip(np.round(v + dl * 255), 0, 255).astype(np.uint8)
        new_hsv_image = cv2.cvtColor(np.stack([h, s, v], axis=2), cv2.COLOR_HSV2RGB)
        alpha_image = prev_image[:, :, 3:]
        return np.concatenate([new_hsv_image, alpha_image], axis=2)
