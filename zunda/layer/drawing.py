from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from zunda.attribute import Attribute, AttributesMixin, AttributeType


class Rectangle(AttributesMixin):

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
        image = np.zeros((H, W, 4), dtype=np.uint8)
        if line_width == 0:
            image[:, :, :] = np.array(color.tolist() + [0], dtype=np.uint8).reshape(1, 1, 4)
        else:
            image[:, :, :] = np.array(line_color.tolist() + [0], dtype=np.uint8).reshape(1, 1, 4)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        x, y = eps + line_width // 2, eps + line_width // 2
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


class Text:

    def __init__(
            self, text: str, font: str, font_size: int,
            text_color: Union[tuple[float, float, float], np.ndarray] = (0., 0., 0.),
            duration: float = 1.):
        self.text = text
        self.font = font
        self.font_size = font_size
        self.text_color = (int(text_color[0]), int(text_color[1]), int(text_color[2]))
        self.duration = duration
        self._font: ImageFont.FreeTypeFont = ImageFont.truetype(self.font, self.font_size)
        self._size: tuple[int, int] = self._font.getsize(self.text)

    def get_key(self, time: float) -> bool:
        return True

    def __call__(self, time: float) -> np.ndarray:
        image = np.zeros((self._size[1], self._size[0], 4), dtype=np.uint8)
        image[:, :, :] = np.array(self.text_color + (0,), dtype=np.uint8).reshape(1, 1, 4)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), self.text, font=self._font, fill=self.text_color)
        return np.asarray(image)
