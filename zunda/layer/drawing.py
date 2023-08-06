from typing import Hashable, Union

import numpy as np
from PIL import Image, ImageDraw

from zunda.attribute import Attribute, AttributeType, normalize_to_tuple


class RectangleLayer:

    def __init__(
            self,
            size: tuple[float, float] = (100., 100.),
            radius: float = 0.,
            color: Union[tuple[float, float, float], np.ndarray] = (0., 0., 0.),
            line_width: float = 0.,
            line_color: Union[tuple[float, float, float], np.ndarray] = (255., 255., 255.),
            duration: float = 1.):
        self.size = Attribute(size, value_type=AttributeType.VECTOR2D)
        self.radius = Attribute(radius, value_type=AttributeType.SCALAR)
        self.color = Attribute(color, value_type=AttributeType.COLOR)
        self.line_width = Attribute(line_width, value_type=AttributeType.SCALAR)
        self.line_color = Attribute(line_color, value_type=AttributeType.COLOR)
        self.duration = duration

    @property
    def attributes(self) -> dict[str, Attribute]:
        return {
            'size': self.size,
            'radius': self.radius,
            'color': self.color,
            'line_width': self.line_width,
            'line_color': self.line_color,
        }

    def get_key(self, time: float) -> Hashable:
        return tuple([normalize_to_tuple(attr(time)) for attr in self.attributes.values()])

    def __call__(self, time: float) -> np.ndarray:
        size = np.round(self.size(time))
        w, h = int(size[0]), int(size[1])
        radius = int(self.radius(time))
        color = np.round(self.color(time))
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        line_width = int(self.line_width(time))
        line_color = np.round(self.line_color(time))
        lr, lg, lb = int(line_color[0]), int(line_color[1]), int(line_color[2])

        eps = 1
        H = h + line_width + 2 * eps
        W = w + line_width + 2 * eps
        image = Image.new("RGBA", (W, H))
        x, y = eps + line_width // 2, eps + line_width // 2
        draw = ImageDraw.Draw(image)
        if radius == 0:
            draw.rectangle(
                (x, y, x + w, y + h), fill=(r, g, b),
                outline=None if line_width == 0 else (lr, lg, lb),
                width=line_width)
        else:
            draw.rounded_rectangle(
                (x, y, x + w, y + h), radius=radius, fill=(r, g, b),
                outline=None if line_width == 0 else (lr, lg, lb),
                width=line_width)
        return np.asarray(image)
