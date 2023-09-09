from __future__ import annotations

import cv2
import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..imgproc import alpha_composite
from ..util import to_rgb


class DropShadow(AttributesMixin):
    """Applies drop shadow to the input image.

    Args:
        radius:
            Radius of Gaussian kernel.
        offset:
            Offset of the shadow in pixels.
        angle:
            Angle of the shadow in degrees.
        color:
            Color of the shadow. It can be specified as a tuple of RGB values or a string of color name.
        opacity:
            Opacity of the shadow in the range ``[0, 1]``.

    Animatable Attributes:
        ``radius``
        ``offset``
        ``angle``
        ``color``
        ``opacity``
"""

    def __init__(
        self,
        radius: float = 0.0,
        offset: float = 0.0,
        angle: float = 45.0,
        color: tuple[int, int, int] | str = (0, 0, 0),
        opacity: float = 0.5
    ) -> None:
        self.radius = Attribute(radius, AttributeType.SCALAR, range=(0., 1e6))
        self.angle = Attribute(angle, AttributeType.SCALAR)
        self.offset = Attribute(offset, AttributeType.SCALAR)
        c = to_rgb(color)
        self.color = Attribute(c, AttributeType.COLOR, range=(0., 255.))
        self.opacity = Attribute(opacity, AttributeType.SCALAR, range=(0., 1.))

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        assert prev_image.ndim == 3
        assert prev_image.shape[2] == 4, f'prev_image must be RGBA image, but {prev_image.shape}'

        radius = float(self.radius(time))
        assert 0 <= radius, f'radius must be nonnegative, but {radius}'

        # Compute ksize and create shadow image
        if 0. < radius:
            k = 2 * int(np.ceil(max(1, (4 * radius - 1) / 2))) + 1
            ksize = (k, k)
            alpha_shadow_image = cv2.GaussianBlur(
                src=np.pad(prev_image[:, :, 3], (ksize, ksize), mode='constant'),
                ksize=ksize, sigmaX=radius, sigmaY=radius)
        else:
            alpha_shadow_image = (self.opacity(time) * prev_image[:, :, 3]).astype(np.uint8)
        rgb_shadow_image = np.full(
            alpha_shadow_image.shape + (3,),
            np.round(self.color(time)).astype(np.uint8), dtype=np.uint8)
        alpha_shadow_image = (self.opacity(time) * alpha_shadow_image).astype(np.uint8)
        shadow_image = np.concatenate([rgb_shadow_image, alpha_shadow_image[:, :, None]], axis=2)

        # Compute offset and paste shadow image
        theta = float(self.angle(time)) * np.pi / 180
        vec = np.round(float(self.offset(time)) * np.array([np.cos(theta), np.sin(theta)])).astype(int)

        Hs, Ws = shadow_image.shape[:2]
        delta = np.abs(vec)
        new_image_shape = (Hs + 2 * delta[1], Ws + 2 * delta[0], 4)
        new_image = np.zeros(new_image_shape, dtype=np.uint8)
        p_shadow = delta + vec
        new_image[p_shadow[1]: (p_shadow[1] + Hs), p_shadow[0]: (p_shadow[0] + Ws)] = shadow_image

        # Put prev_image on the center of new_image
        p_image = (new_image_shape[1] - prev_image.shape[1]) // 2, (new_image_shape[0] - prev_image.shape[0]) // 2
        return alpha_composite(new_image, prev_image, p_image)
