from typing import Union

import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..util import hex_to_rgb


class Gradation(AttributesMixin):

    def __init__(
        self,
        size: tuple[int, int] = (100, 100),
        start_point: tuple[float, float] = (0., 0.),
        end_point: tuple[float, float] = (100., 100.,),
        start_color: Union[tuple[int, int, int], str] = (0, 0, 0),
        end_color: Union[tuple[int, int, int], str] = (255, 255, 255),
        gradation_type: str = 'linear',
        duration: float = 1.0,
    ) -> None:
        self.size = size
        self.duration = duration
        self.start_point = Attribute(start_point, AttributeType.VECTOR2D)
        self.end_point = Attribute(end_point, AttributeType.VECTOR2D)
        cs = hex_to_rgb(start_color) if isinstance(start_color, str) else start_color
        ce = hex_to_rgb(end_color) if isinstance(end_color, str) else end_color
        self.start_color = Attribute(cs, AttributeType.COLOR, range=(0., 255.))
        self.end_color = Attribute(ce, AttributeType.COLOR, range=(0., 255.))
        if gradation_type not in ('linear', 'radial'):
            raise ValueError(f"Invalid gradation_type: {gradation_type}. 'linear' or 'radial' is expected.")
        self.gradation_type = gradation_type

    def __call__(self, time: float) -> np.ndarray:
        width, height = self.size
        inds = np.mgrid[:height, :width]
        ps = self.start_point(time)[::-1]
        pe = self.end_point(time)[::-1]
        v = pe - ps
        if self.gradation_type == 'linear':
            p = ((inds - ps.reshape(2, 1, 1)) * v.reshape(2, 1, 1)).sum(axis=0)
        elif self.gradation_type == 'radial':
            p = ((inds - ps.reshape(2, 1, 1)) ** 2).sum(axis=0)
        p = np.sqrt(np.clip(p / (v ** 2).sum(), 0., 1.))

        cs = self.start_color(time)
        ce = self.end_color(time)
        color = np.round(p * ce.reshape(3, 1, 1) + (1 - p) * cs.reshape(3, 1, 1))
        color = color.astype(np.uint8).transpose(1, 2, 0)
        color = np.concatenate([color, np.full((height, width, 1), 255, dtype=np.uint8)], axis=2)
        return color
