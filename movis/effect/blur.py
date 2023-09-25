import cv2
import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..enum import BlendingMode
from ..imgproc import alpha_composite


class GaussianBlur(AttributesMixin):
    """Gaussian blur effect.

    It applies Gaussian blur to the input image.

    .. note::
        The resolution of the output image is greater than
        that of the input image to preserve the blurring of the edges.

    Args:
        radius:
            Radius of Gaussian kernel.

    Animatable Attributes:
        ``radius``
    """

    def __init__(self, radius: float):
        self.radius = Attribute(radius, AttributeType.SCALAR, range=(0., 1e6))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'
        radius = float(self.radius(time))
        if radius == 0:
            return prev_image
        assert 0 < radius, f'radius must be nonnegative, but {radius}'

        k = 2 * int(np.ceil(max(1, (4 * radius - 1) / 2))) + 1
        ksize = (k, k)
        rgb_image = np.pad(prev_image[:, :, :3], (ksize, ksize, (0, 0)), mode='edge')
        alpha_image = np.pad(prev_image[:, :, 3:], (ksize, ksize, (0, 0)), mode='constant')
        return cv2.GaussianBlur(
            src=np.concatenate([rgb_image, alpha_image], axis=2),
            ksize=ksize, sigmaX=radius, sigmaY=radius)


class Glow(AttributesMixin):
    """Glow effect.

    It applies Gaussian blur to the input image and adds it to the original with the specified strength.

    Args:
        radius:
            Radius of Gaussian kernel.
        strength:
            Strength of the glow effect.

    Animatable Attributes:
        ``radius``
        ``strength``
    """

    def __init__(self, radius: float, strength: float = 1.0):
        self.radius = Attribute(radius, AttributeType.SCALAR, range=(0., 1e6))
        self.strength = Attribute(strength, AttributeType.SCALAR, range=(0., 100.))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        radius = float(self.radius(time))
        if radius == 0.:
            return prev_image

        assert 0 < radius, f'radius must be positive, but {radius}'
        k = 2 * int(np.ceil(max(1, (4 * radius - 1) / 2))) + 1
        ksize = (k, k)
        rgb_image = np.pad(prev_image[:, :, :3], (ksize, ksize, (0, 0)), mode='edge')
        alpha_image = np.pad(prev_image[:, :, 3:], (ksize, ksize, (0, 0)), mode='constant')
        pad_image = np.concatenate([rgb_image, alpha_image], axis=2)
        blurred_image = cv2.GaussianBlur(
            src=pad_image,
            ksize=ksize, sigmaX=radius, sigmaY=radius)
        strength = float(self.strength(time))
        bimg_rgb = blurred_image[:, :, :3].astype(np.float32)
        bimg_rgb = np.clip(strength * blurred_image[:, :, :3].astype(np.float32), 0, 255).astype(np.uint8)
        blurred_image = np.concatenate([bimg_rgb, blurred_image[:, :, 3:]], axis=2)
        return alpha_composite(
            pad_image, blurred_image, blending_mode=BlendingMode.LINEAR_DODGE)
