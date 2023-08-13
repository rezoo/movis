import cv2
import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType


class GaussianBlur(AttributesMixin):

    def __init__(self, radius: float):
        self.radius = Attribute(radius, AttributeType.SCALAR)

    def __call__(self, time: float, prev_image: np.ndarray) -> np.ndarray:
        def get_ksize(sigma):
            return 2 * int(np.ceil(max(1, (3 * sigma - 1) / 2))) + 1
        radius = float(self.radius(time))
        assert 0 < radius, f'radius must be positive, but {radius}'
        ksize = get_ksize(radius)
        return cv2.GaussianBlur(src=prev_image, ksize=ksize, sigmaX=radius, sigmaY=radius)
