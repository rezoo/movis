from __future__ import annotations

import numpy as np
from PySide6.QtGui import (QColor, QImage, QLinearGradient, QPainter,
                           QRadialGradient)

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..imgproc import qimage_to_numpy
from ..util import to_rgb


class Gradient(AttributesMixin):
    """A layer that generates a gradient image.

    Args:
        size:
            the size of the generated image. Defaults to ``(100, 100)``.
        start_point:
            the start point of the gradient. Defaults to ``(0., 0.)``.
        end_point:
            the end point of the gradient. Defaults to ``(100., 100.)``.
        start_color:
            the start color of the gradient. Defaults to ``(0, 0, 0)``.
            the color can be specified as a tuple of ``(R, G, B)``
            or a string (`e.g.,` ``"#ff0000"``, or ``"red"``).
        end_color:
            the end color of the gradient. Defaults to ``(255, 255, 255)``.
        gradient_type:
            the type of the gradient. "linear" or "radial" is expected. Defaults to ``"linear"``.
        duration:
            the duration of the layer. Defaults to ``1e6``.

    Animatable Attributes:
        ``start_point``
        ``end_point``
        ``start_color``
        ``end_color``
    """

    def __init__(
        self,
        size: tuple[int, int] = (100, 100),
        start_point: tuple[float, float] = (0., 0.),
        end_point: tuple[float, float] = (100., 100.,),
        start_color: tuple[int, int, int] | str = (0, 0, 0),
        end_color: tuple[int, int, int] | str = (255, 255, 255),
        gradient_type: str = 'linear',
        duration: float = 1e6,
    ) -> None:
        self.size = size
        self.duration = duration
        self.start_point = Attribute(start_point, AttributeType.VECTOR2D)
        self.end_point = Attribute(end_point, AttributeType.VECTOR2D)
        cs = to_rgb(start_color)
        ce = to_rgb(end_color)
        self.start_color = Attribute(cs, AttributeType.COLOR, range=(0., 255.))
        self.end_color = Attribute(ce, AttributeType.COLOR, range=(0., 255.))
        if gradient_type not in ('linear', 'radial'):
            raise ValueError(f"Invalid gradation_type: {gradient_type}. 'linear' or 'radial' is expected.")
        self.gradient_type = gradient_type

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0 or time >= self.duration:
            return None
        width, height = self.size
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        ps = self.start_point(time)
        pe = self.end_point(time)
        cs = np.round(self.start_color(time)).astype(int)
        ce = np.round(self.end_color(time)).astype(int)
        if self.gradient_type == 'linear':
            grad_linear = QLinearGradient(
                float(ps[0]) + 0.5, float(ps[1]) + 0.5, float(pe[0]) - 0.5, float(pe[1]) - 0.5)
            grad_linear.setColorAt(0, QColor(cs[2], cs[1], cs[0], 255))
            grad_linear.setColorAt(1, QColor(ce[2], ce[1], ce[0], 255))
            painter.fillRect(0, 0, width, height, grad_linear)
        elif self.gradient_type == 'radial':
            radius = np.sqrt(((ps - pe) ** 2).sum())
            grad_radial = QRadialGradient(float(ps[0]) + 0.5, float(ps[1]) + 0.5, float(radius))
            grad_radial.setColorAt(0, QColor(cs[2], cs[1], cs[0], 255))
            grad_radial.setColorAt(1, QColor(ce[2], ce[1], ce[0], 255))
            painter.fillRect(0, 0, width, height, grad_radial)
        painter.end()
        return qimage_to_numpy(image)


class Stripe(AttributesMixin):
    """A layer that generates a stripe pattern.

    Args:
        size:
            the size of the generated image. Defaults to ``(100, 100)``.
        angle:
            the angle of the stripe pattern in degrees. Defaults to ``45.0``.
        color1:
            the first color of the stripe pattern. Defaults to ``(0, 0, 0)``.
            the color can be specified as a tuple of ``(R, G, B)``,
            or a string (`e.g.,` ``"#ff0000"``, or ``"red"``).
        color2:
            the second color of the stripe pattern. Defaults to ``(255, 255, 255)``.
        opacity1:
            the opacity of the first color. Defaults to ``1.0``.
        opacity2:
            the opacity of the second color. Defaults to ``1.0``.
        total_width:
            the total width of the stripe pattern. The stripe pattern repeats every ``total_width``.
            Defaults to ``64.0``.
        phase:
            the phase of the stripe pattern in degrees. Defaults to ``0.0``.
        ratio:
            the ratio of the first color. Defaults to ``0.5``.
        duration:
            the duration of the layer. Defaults to ``1e6``.

    Animatable Attributes:
        ``angle``
        ``color1``
        ``color2``
        ``opacity1``
        ``opacity2``
        ``total_width``
        ``phase``
        ``ratio``
    """
    def __init__(
        self,
        size: tuple[int, int] = (100, 100),
        angle: float = 45.,
        color1: tuple[int, int, int] | str = (0, 0, 0),
        color2: tuple[int, int, int] | str = (255, 255, 255),
        opacity1: float = 1.0,
        opacity2: float = 1.0,
        total_width: float = 64.,
        phase: float = 0.,
        ratio: float = 0.5,
        duration: float = 1e6,
    ) -> None:
        self.size = size
        self.duration = duration
        self.angle = Attribute(angle, AttributeType.ANGLE)
        c1 = to_rgb(color1)
        c2 = to_rgb(color2)
        self.color1 = Attribute(c1, AttributeType.COLOR, range=(0., 255.))
        self.color2 = Attribute(c2, AttributeType.COLOR, range=(0., 255.))
        self.opacity1 = Attribute(opacity1, AttributeType.SCALAR, range=(0., 1.0))
        self.opacity2 = Attribute(opacity2, AttributeType.SCALAR, range=(0., 1.0))
        self.total_width = Attribute(total_width, AttributeType.SCALAR, range=(0., 1e6))
        self.phase = Attribute(phase, AttributeType.SCALAR)
        self.ratio = Attribute(ratio, AttributeType.SCALAR, range=(0., 1.0))

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0 or time >= self.duration:
            return None
        width, height = self.size
        ratio = float(self.ratio(time))
        c1 = np.concatenate([
            np.round(self.color1(time)).reshape(3, 1, 1),
            np.full((1, 1, 1), np.round(255 * float(self.opacity1(time))))], axis=0)
        c2 = np.concatenate([
            np.round(self.color2(time)).reshape(3, 1, 1),
            np.full((1, 1, 1), np.round(255 * float(self.opacity2(time))))], axis=0)
        if ratio <= 0.0:
            c1_img = np.broadcast_to(c1, (4, height, width)).transpose(1, 2, 0)
            return c1_img.astype(np.uint8)
        elif ratio >= 1.0:
            c2_img = np.broadcast_to(c2, (4, height, width)).transpose(1, 2, 0)
            return c2_img.astype(np.uint8)
        center = np.array([height / 2, width / 2])[:, None, None]
        inds = np.mgrid[:height, :width] - center
        theta = float(self.angle(time)) / 180.0 * np.pi
        phase = float(self.phase(time))
        stripe_width = float(self.total_width(time))

        eps = 1e-2
        v = np.array([np.sin(theta), np.cos(theta)], dtype=np.float64)
        p = (v.reshape(2, 1, 1) * inds).sum(axis=0, keepdims=True) / stripe_width + phase
        p = _fract(p)
        p = _smoothstep(max(ratio - eps, 0.0), min(ratio + eps, 1.0), p)
        color = p * np.broadcast_to(c1, (4, height, width)) + (1 - p) * np.broadcast_to(c2, (4, height, width))
        color = color.transpose(1, 2, 0).astype(np.uint8)
        return color


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fract(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)
