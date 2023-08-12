import sys
from typing import Union, Callable

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt, QRectF
from PySide6.QtGui import QImage, QColor, QPainter, QBrush, QPen, QFont, QFontMetrics, QFontDatabase

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
        size = [float(x) for x in self.size(time)]
        w, h = float(size[0]), float(size[1])
        radius = float(self.radius(time))
        color = np.round(self.color(time))
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        line_width = float(self.line_width(time))
        line_color = np.round(self.line_color(time))
        lr, lg, lb = int(line_color[0]), int(line_color[1]), int(line_color[2])

        eps = 0
        H = np.floor(h + line_width + 2 * eps)
        W = np.floor(w + line_width + 2 * eps)
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        if line_width == 0:
            image.fill(QColor(b, g, r, 255))
        else:
            image.fill(QColor(lb, lg, lr, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        brush = QBrush(QColor(b, g, r, 255))
        painter.setBrush(brush)
        pen = QPen(QColor(lb, lg, lr, 255), line_width)
        painter.setPen(pen)

        rect = QRectF(eps + line_width / 2, eps + line_width / 2, w, h)
        painter.drawRoundedRect(rect, radius, radius, mode=Qt.SizeMode.AbsoluteSize)
        painter.end()

        assert image.format() == QImage.Format.Format_ARGB32
        ptr = image.bits()
        array_shape = (image.height(), image.width(), 4)
        return np.array(ptr).reshape(array_shape)


class Text:

    def __init__(
            self, text: Union[str, Callable[[float], str]], font: str, font_size: int,
            text_color: Union[tuple[float, float, float], np.ndarray] = (0., 0., 0.),
            duration: float = 1.):
        self.text = text
        self.font = font
        self.font_size = font_size
        self.text_color = (int(text_color[0]), int(text_color[1]), int(text_color[2]))
        self.duration = duration
        if QCoreApplication.instance() is None:
            self._app = QApplication(sys.argv[:1])
        self._fontid = QFontDatabase.addApplicationFont(self.font)
        self._font_family = QFontDatabase.applicationFontFamilies(self._fontid)
        self._qfont = QFont(self._font_family[0], self.font_size)

    def get_text(self, time: float = 0.) -> str:
        if isinstance(self.text, str):
            return self.text
        elif callable(self.text):
            return self.text(time)
        else:
            raise ValueError(f"Invalid text type: {type(self.text)}")

    def get_size(self, time: float = 0.) -> tuple[int, int]:
        metrics = QFontMetrics(self._qfont)
        text = self.get_text(time)
        rect = metrics.boundingRect(text)
        text_width = rect.width()
        text_height = rect.height()
        return (text_width, text_height)

    def get_key(self, time: float) -> str:
        return self.get_text(time)

    def __call__(self, time: float) -> np.ndarray:
        size = self.get_size(time)
        eps = 5
        image = QImage(size[0] + 2 * eps, size[1] + 2 * eps, QImage.Format.Format_ARGB32)
        r, g, b = self.text_color
        image.fill(QColor(b, g, r, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.setPen(QColor(b, g, r, 255))
        painter.setFont(self._qfont)

        painter.drawText(0, eps + size[1], self.get_text(time))
        painter.end()

        assert image.format() == QImage.Format.Format_ARGB32
        ptr = image.bits()
        array_shape = (image.height(), image.width(), 4)
        return np.array(ptr).reshape(array_shape)
