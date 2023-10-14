from __future__ import annotations

import sys
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from PySide6.QtCore import QCoreApplication, QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetrics,
                           QImage, QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import QApplication

from movis.imgproc import qimage_to_numpy

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..enum import TextAlignment
from ..util import to_rgb
from .mixin import TimelineMixin


class FillProperty:
    """A property for filling a shape.

    Args:
        color:
            The color of the shape with a tuple of ``(r, g, b)``
            or a string representing a color name (`e.g.,` ``"#ff0000" or "red"``).
        opacify:
            The opacity of the shape in the range of ``[0, 1]``.
    """

    def __init__(self, color: tuple[int, int, int] | str, opacify: float = 1.):
        self._color: tuple[int, int, int] = to_rgb(color)
        self._opacity: float = float(opacify)

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    @property
    def opacity(self) -> float:
        return self._opacity


class StrokeProperty:
    """A property for stroking a shape.

    Args:
        color:
            The color of the shape with a tuple of ``(r, g, b)``
            or a string representing a color name (`e.g.,` ``"#ff0000" or "red"``).
        width:
            The width of the stroke.
        opacity:
            The opacity of the shape in the range of ``[0, 1]``.
    """

    def __init__(self, color: tuple[int, int, int] | str, width: float = 1., opacity: float = 1.):
        self._color: tuple[int, int, int] = to_rgb(color)
        self._width: float = float(width)
        self._opacity: float = float(opacity)

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    @property
    def width(self) -> float:
        return self._width

    @property
    def opacity(self) -> float:
        return self._opacity


class Line(AttributesMixin):
    """Draw a line from ``start`` to ``end``.

    Args:
        size:
            The size of the canvas with a tuple of ``(width, height)``.
        start:
            The start point of the line with a tuple of ``(x, y)``.
        end:
            The end point of the line with a tuple of ``(x, y)``.
        color:
            The color of the line with a tuple of ``(r, g, b)``
            or a string representing a color name (`e.g.,` ``"#ff0000" or "red"``).
        width:
            The width of the line.
        duration:
            The duration for which the line should be displayed.

    Animateable Attributes:
        ``trim_start``
            The start point of the line to be drawn in the range of ``[0, 1]``.
            The default value is ``0``.
        ``trim_end``
            The end point of the line to be drawn in the range of ``[0, 1]``.
            The default value is ``1``.
        ``start`` ``end`` ``color`` ``width``
            These attributes can be animated as well.
    """
    def __init__(
        self,
        size: tuple[int, int] = (100, 100),
        start: tuple[float, float] | np.ndarray = (0., 0.),
        end: tuple[float, float] | np.ndarray = (100., 100.),
        color: tuple[int, int, int] | str = (255, 255, 255),
        width: float = 1.,
        duration: float = 1e6,
    ) -> None:
        self.size = size
        self.start = Attribute(start, value_type=AttributeType.VECTOR2D)
        self.end = Attribute(end, value_type=AttributeType.VECTOR2D)
        self.color = Attribute(to_rgb(color), value_type=AttributeType.COLOR, range=(0., 255.))
        self.width = Attribute(width, value_type=AttributeType.SCALAR, range=(0., 1e6))
        self.trim_start = Attribute(0., value_type=AttributeType.SCALAR, range=(0., 1.))
        self.trim_end = Attribute(1., value_type=AttributeType.SCALAR, range=(0., 1.))
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, time: float) -> np.ndarray | None:
        p0 = self.start(time)
        p1 = self.end(time)
        trim_start = self.trim_start(time)[0]
        trim_end = self.trim_end(time)[0]
        assert 0. <= trim_start <= trim_end <= 1.
        p_start = p0 + trim_start * (p1 - p0)
        p_end = p0 + trim_end * (p1 - p0)
        r, g, b = tuple(np.round(self.color(time)).astype(int))

        W, H = self.size
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(b, g, r, 255), self.width(time)[0]))
        painter.drawLine(QPointF(*p_start), QPointF(*p_end))

        painter.end()
        return qimage_to_numpy(image)


class Rectangle(AttributesMixin):
    """Draw a rectangle with rounded corners.

    Args:
        size:
            The size of the rectangle with a tuple of ``(width, height)``.
        radius:
            The radius of the rounded corners. The default value is ``0``.
        color:
            The color of the rectangle with a tuple of ``(r, g, b)`` or a string (e.g., ``"#ff0000" or "red"``).
            If ``None``, this layer uses the ``contents`` argument to draw the rectangle.
        contents:
            A sequence of ``FillProperty`` or ``StrokeProperty`` objects,
            which will be drawn on top of the rectangle.
            If an empty sequence is given, the rectangle will be filled with the ``color`` argument.
        duration:
            The duration for which the rectangle should be displayed.

    Animateable Attributes:
        ``size``
        ``radius``
    """
    def __init__(
        self,
        size: tuple[float, float] = (100., 100.),
        radius: float = 0.,
        color: tuple[int, int, int] | str | None = None,
        contents: Sequence[FillProperty | StrokeProperty] = (),
        duration: float = 1e6
    ) -> None:
        self.size = Attribute(size, value_type=AttributeType.VECTOR2D, range=(0., 1e6))
        self.radius = Attribute(radius, value_type=AttributeType.SCALAR, range=(0., 1e6,))
        if color is None:
            self.contents = contents
        else:
            self.contents = (FillProperty(color=to_rgb(color)),)
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, time: float) -> np.ndarray | None:
        if len(self.contents) == 0:
            return None
        size = [float(x) for x in self.size(time)]
        w, h = float(size[0]), float(size[1])
        radius = float(self.radius(time))

        eps = 1.
        max_stroke = _get_max_stroke(self.contents)
        W = np.floor(w + max_stroke + 2 * eps)
        H = np.floor(h + max_stroke + 2 * eps)
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        max_color = _get_max_color(self.contents)
        if max_color is None:
            image.fill(QColor(0, 0, 0, 0))
        else:
            image.fill(QColor(*max_color, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(eps + max_stroke / 2, eps + max_stroke / 2, w, h)
        for c in self.contents:
            if isinstance(c, FillProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                painter.setBrush(QBrush(QColor(b, g, r, a)))
                painter.setPen(QPen(QColor(b, g, r, 0)))
                painter.drawRoundedRect(rect, radius, radius, mode=Qt.SizeMode.AbsoluteSize)
            elif isinstance(c, StrokeProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                painter_path = QPainterPath()
                painter_path.addRoundedRect(rect, radius, radius, mode=Qt.SizeMode.AbsoluteSize)
                painter.setBrush(QBrush(QColor(b, g, r, 0)))
                painter.setPen(QPen(QColor(b, g, r, a), c.width))
                painter.drawPath(painter_path)
            else:
                raise ValueError(f"Invalid content type: {type(c)}")
        painter.end()
        return qimage_to_numpy(image)


class Ellipse(AttributesMixin):
    """Draw an ellipse.

    Args:
        size:
            The size of the ellipse with a tuple of ``(width, height)``.
        color:
            The color of the ellipse with a tuple of ``(r, g, b)``,
            or a string representing a color name (`e.g.,` ``"#ff0000" or "red"``).
            If ``None``, this layer uses the ``contents`` argument to draw the ellipse.
        contents:
            A sequence of ``FillProperty`` or ``StrokeProperty`` objects,
            which will be drawn on top of the ellipse.
            If an empty sequence is given, the ellipse will be filled with the ``color`` argument.
        duration:
            The duration for which the ellipse should be displayed.

    Animateable Attributes:
        ``size``
    """
    def __init__(
        self,
        size: tuple[float, float] = (100., 100.),
        color: tuple[int, int, int] | str | None = None,
        contents: Sequence[FillProperty | StrokeProperty] = (),
        duration: float = 1e6
    ) -> None:
        self.size = Attribute(size, value_type=AttributeType.VECTOR2D, range=(0., 1e6))
        if color is None:
            self.contents = contents
        else:
            self.contents = (FillProperty(color=to_rgb(color)),)
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, time: float) -> np.ndarray | None:
        if len(self.contents) == 0:
            return None
        size = [float(x) for x in self.size(time)]
        w, h = float(size[0]), float(size[1])

        eps = 1.
        max_stroke = _get_max_stroke(self.contents)
        W = np.floor(w + max_stroke + 2 * eps)
        H = np.floor(h + max_stroke + 2 * eps)
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        max_color = _get_max_color(self.contents)
        if max_color is None:
            image.fill(QColor(0, 0, 0, 0))
        else:
            image.fill(QColor(*max_color, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(eps + max_stroke / 2, eps + max_stroke / 2, w, h)
        for c in self.contents:
            if isinstance(c, FillProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                painter.setBrush(QBrush(QColor(b, g, r, a)))
                painter.setPen(QPen(QColor(b, g, r, 0)))
                painter.drawEllipse(rect)
            elif isinstance(c, StrokeProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                painter_path = QPainterPath()
                painter_path.addEllipse(rect)
                painter.setPen(QPen(QColor(b, g, r, a), c.width))
                painter.setBrush(QBrush(QColor(b, g, r, 0)))
                painter.drawPath(painter_path)
            else:
                raise ValueError(f"Invalid content type: {type(c)}")
        painter.end()
        return qimage_to_numpy(image)


class _TextWithTime(TimelineMixin):
    def __init__(self, start_times: Sequence[float], end_times: Sequence[float], texts: Sequence[str]):
        super().__init__(start_times, end_times)
        self.texts = texts

    def __call__(self, time: float) -> str:
        idx = self.get_state(time)
        if idx >= 0:
            return self.texts[idx]
        else:
            return ''


class Text(AttributesMixin):
    """Draw a text.

    Args:
        text:
            the text to be drawn. It can be a string or a callable object.
            If it is a callable object, it must accept a float value representing the time,
            and return a string.
        font_size:
            the font size of the text.
        font_family:
            the font family of the text. It must be one of the available fonts (`e.g.,` ``"Helvetica"``).
            To see the list of available fonts, run ``movis.layer.Text.available_fonts()``.
        font_style:
            the font style of the text. It must be one of the available styles of the given font family
            (`e.g.,` ``"Bold"``). To see the list of available styles,
            un ``movis.layer.Text.available_styles(font_name)``.
        color:
            the color of the text with a tuple of ``(r, g, b)``,
            or a string representing a color name (`e.g.,` ``"#ff0000" or "red"``).
            If ``None``, this layer uses the ``contents`` argument to draw the text.
        contents:
            A sequence of ``FillProperty`` or ``StrokeProperty`` objects,
            which will be drawn on top of the text.
            If an empty sequence is given, the text will be filled with the ``color`` argument.
        line_spacing:
            the line spacing of the text. If ``None``, the line spacing is automatically determined.
        text_alignment:
            the text alignment. If string is given, it must be one of ``"left"``, ``"center"``, or ``"right"``.
            It also accepts a ``TextAlignment`` enum.
        duration:
            the duration for which the text should be displayed.

        Animateable Attributes:
            ``font_size``
    """

    @staticmethod
    def available_fonts() -> Sequence[str]:
        """Returns the list of available fonts."""
        if QCoreApplication.instance() is None:
            QApplication(sys.argv[:1])
        return QFontDatabase.families()

    @staticmethod
    def available_styles(font_name: str) -> Sequence[str]:
        """Returns the list of available styles of the given font family."""
        if QCoreApplication.instance() is None:
            QApplication(sys.argv[:1])
        return QFontDatabase.styles(font_name)

    @classmethod
    def from_timeline(
        cls,
        start_times: Sequence[float],
        end_times: Sequence[float],
        texts: Sequence[str],
        **kwargs
    ) -> 'Text':
        """Create a text layer from a timeline.

        This method is useful when you want to display different texts at different times
        (e.g., displaying a subtitle). Note that other arguments are the same as the constructor.

        Args:
            start_times:
                A sequence of start times of the texts.
            end_times:
                A sequence of end times of the texts.
            texts:
                A sequence of texts to be displayed.

        Returns:
            A new `Text` object.
        """
        assert len(start_times) == len(texts)

        kwargs['duration'] = max(end_times)
        return cls(text=_TextWithTime(start_times, end_times, texts), **kwargs)

    def __init__(
        self,
        text: str | Callable[[float], str],
        font_size: float,
        font_family: str = 'Sans Serif',
        font_style: str | None = None,
        color: tuple[int, int, int] | str | None = None,
        contents: Sequence[FillProperty | StrokeProperty] = (),
        line_spacing: int | None = None,
        text_alignment: TextAlignment | str = TextAlignment.CENTER,
        duration: float = 1e6
    ) -> None:
        self._text = text
        self._font_family = font_family
        self._font_style = font_style
        self.font_size = Attribute(font_size, value_type=AttributeType.SCALAR, range=(0., 1e6))
        if color is None:
            self._contents = tuple(contents)
        else:
            self._contents = (FillProperty(color=to_rgb(color)),)
        self._line_spacing = line_spacing
        self._text_alignment = TextAlignment.from_string(text_alignment) \
            if isinstance(text_alignment, str) else text_alignment
        self._duration = duration
        if QCoreApplication.instance() is None:
            QApplication(sys.argv[:1])
        self._init_app = True

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_init_app'] = False
        return state

    @property
    def text(self) -> str | Callable[[float], str]:
        return self._text

    @property
    def font_family(self) -> str:
        return self._font_family

    @property
    def font_style(self) -> str | None:
        return self._font_style

    @property
    def contents(self) -> tuple[FillProperty | StrokeProperty, ...]:
        return self._contents

    @property
    def line_spacing(self) -> int | None:
        return self._line_spacing

    @property
    def text_alignment(self) -> TextAlignment:
        return self._text_alignment

    @property
    def duration(self) -> float:
        return self._duration

    def get_text(self, time: float = 0.) -> str:
        """Returns the text to be drawn at the given time."""
        if isinstance(self.text, str):
            return self.text
        elif callable(self.text):
            return self.text(time)
        else:
            raise ValueError(f"Invalid text type: {type(self.text)}")

    def _get_qfont(self, time: float) -> QFont:
        if self.font_style is None:
            return QFont(self.font_family, round(float(self.font_size(time))))
        else:
            return QFontDatabase.font(
                self.font_family, self.font_style, round(float(self.font_size(time))))

    def get_size(self, time: float = 0.) -> tuple[int, int]:
        """Returns the size of the text at the given time.

        .. note::
            The returned size is the size of the text drawn on the canvas,
            and it may be different from the size of the text itself.

        Args:
            time:
                The time at which the size of the text is measured.
        """
        qfont = self._get_qfont(time)
        metrics = QFontMetrics(qfont)
        text = self.get_text(time)
        lines = text.split('\n')
        W, H = 0, 0
        for i, line in enumerate(lines):
            rect = metrics.boundingRect(line)
            W = max(W, rect.width() + rect.x())
            if self.line_spacing is None or i == len(lines) - 1:
                H += (rect.height() - rect.y())
            else:
                H += self.line_spacing
        return (W, H)

    def get_key(self, time: float) -> tuple[str, Hashable]:
        """Returns the state of the layer at the given time."""
        key = super().get_key(time)
        return (self.get_text(time), key)

    def _get_current_cursor_position(
        self, metrics: QFontMetrics, line: str, cursor_y: float,
        lineno: int, width: float
    ) -> tuple[float, float]:
        rect = metrics.boundingRect(line)
        if lineno == 0:
            cursor_y += rect.height()
        elif self.line_spacing is None:
            cursor_y += (rect.height() - rect.y())
        else:
            cursor_y += self.line_spacing

        if self.text_alignment == TextAlignment.LEFT:
            cursor_x = 0.
        elif self.text_alignment == TextAlignment.CENTER:
            cursor_x = (width - rect.width() - rect.x()) / 2
        elif self.text_alignment == TextAlignment.RIGHT:
            cursor_x = width - rect.width() - rect.x()
        else:
            raise ValueError(f"Invalid text alignment: {self.text_alignment}")
        return cursor_x, cursor_y

    def __call__(self, time: float) -> np.ndarray | None:
        if len(self.contents) == 0:
            return None
        text = self.get_text(time)
        if text is None or text == '':
            return None
        if not self._init_app:
            if QCoreApplication.instance() is None:
                QApplication(sys.argv[:1])
            self._init_app = True
        size = self.get_size(time)
        w, h = float(size[0]), float(size[1])

        # XXX: The returned rectangle and the area of the text
        # actually drawn seem to be different. We will keep
        # more margin in the interim
        margin = 40.
        max_stroke = _get_max_stroke(self.contents)
        W = np.floor(w + max_stroke + 2 * margin)
        H = np.floor(h + max_stroke + 2 * margin)
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        max_color = _get_max_color(self.contents)
        if max_color is None:
            image.fill(QColor(0, 0, 0, 0))
        else:
            image.fill(QColor(*max_color, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        qfont = self._get_qfont(time)
        metrics = QFontMetrics(qfont)
        painter.setFont(qfont)
        lines = text.split('\n')
        for c in self.contents:
            if isinstance(c, FillProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                painter.setPen(QColor(b, g, r, a))
                cursor_y = margin
                for i, line in enumerate(lines):
                    cursor_x, cursor_y = self._get_current_cursor_position(
                        metrics, line, cursor_y, lineno=i, width=w)
                    painter.drawText(QPointF(max_stroke + margin + cursor_x, cursor_y), line)
            elif isinstance(c, StrokeProperty):
                r, g, b = c.color
                a = round(255 * c.opacity)
                pen = QPen(
                    QColor(b, g, r, a), c.width, Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter_path = QPainterPath()
                cursor_y = margin
                for i, line in enumerate(lines):
                    cursor_x, cursor_y = self._get_current_cursor_position(
                        metrics, line, cursor_y, lineno=i, width=w)
                    painter_path.addText(QPointF(max_stroke + margin + cursor_x, cursor_y), qfont, line)
                painter.drawPath(painter_path)
        painter.end()
        array = qimage_to_numpy(image)
        return _clip_image(array)


def _clip_image(image: np.ndarray) -> np.ndarray:
    assert image.ndim == 3
    assert image.shape[2] == 4
    non_empty_pixels = np.all(image != np.array([0, 0, 0, 0]), axis=-1)
    non_empty_row_indices, non_empty_col_indices = np.where(non_empty_pixels)
    if non_empty_row_indices.size == 0 or non_empty_col_indices.size == 0:
        return image
    top, bottom = np.min(non_empty_row_indices), np.max(non_empty_row_indices)
    left, right = np.min(non_empty_col_indices), np.max(non_empty_col_indices)
    clipped_image = image[top: bottom + 1, left: right + 1]
    return clipped_image


def _get_max_stroke(contents: Sequence[FillProperty | StrokeProperty]) -> float:
    strokes = [c.width for c in contents if isinstance(c, StrokeProperty)]
    return float(max(strokes)) if 0 < len(strokes) else 0.


def _get_max_color(
    contents: Sequence[FillProperty | StrokeProperty]
) -> tuple[int, int, int] | None:
    strokes = [(c.width, c.color) for c in contents if isinstance(c, StrokeProperty)]
    if len(strokes) == 0:
        fills = [c.color for c in contents if isinstance(c, FillProperty)]
        return fills[-1] if 0 < len(fills) else None
    else:
        return max(strokes, key=lambda x: x[0])[1]
