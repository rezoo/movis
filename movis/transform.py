from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from movis.enum import BlendingMode, Direction

from .attribute import Attribute, AttributeType


class TransformValue(NamedTuple):
    """A named tuple that encapsulates various transformation properties that can be applied to a layer.

    The properties of `TransformValue` include the anchor point, position, scale, rotation, opacity, and origin point.

    Attributes:
        anchor_point:
            A tuple of two floats representing the anchor point ``(x, y)`` of an object. Defaults to ``(0.0, 0.0)``.
        position:
            A tuple of two floats representing the position ``(x, y)`` of an object. Defaults to ``(0.0, 0.0)``.
        scale:
            A tuple of two floats representing the scale ``(x, y)`` of an object. Defaults to ``(1.0, 1.0)``.
        rotation:
            A float value representing the rotation angle in degrees. Defaults to ``0.0``.
        opacity:
            A float value representing the opacity of an object. Must be in the range ``[0, 1]``. Defaults to ``1.0``.
        origin_point:
            An enum value from Direction representing the origin point for transformations.
            Defaults to ``Direction.CENTER``.
        origin_point:
            An enum value from Direction representing the blending mode of the layer.
            Defaults to ``BlendingMode.NORMAL``.
    """

    anchor_point: tuple[float, float] = (0.0, 0.0)
    position: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    opacity: float = 1.0
    origin_point: Direction = Direction.CENTER
    blending_mode: BlendingMode = BlendingMode.NORMAL


class Transform:
    """A class responsible for encapsulating the various transformation attributes for a layer.

    It utilizes ``Attribute`` class to enforce types and optionally ranges for each attribute.

    Args:
        position:
            A float, tuple of floats, or numpy ndarray representing the position ``(x, y)`` of an object.
            Defaults to ``(0.0, 0.0)``.
        scale:
            A float, tuple of floats, or numpy ndarray representing the scale ``(x, y)`` of an object.
            Defaults to ``(1.0, 1.0)``.
        rotation:
            A float value representing the rotation angle in degrees. Defaults to ``0.0``.
        opacity:
            A float value representing the opacity of an object. Must be in the range ``[0, 1]``. Defaults to ``1.0``.
        anchor_point:
            A float, tuple of floats, or ``numpy.ndarray`` representing the anchor point ``(x, y)`` of an object.
            Defaults to ``(0.0, 0.0)``.
        origin_point:
            An enum value from Direction or a string representing the origin point for transformations.
            Defaults to ``Direction.CENTER``.
    """

    def __init__(
        self,
        position: float | tuple[float, float] | np.ndarray = (0.0, 0.0),
        scale: float | tuple[float, float] | np.ndarray = (1.0, 1.0),
        rotation: float = 0.0,
        opacity: float = 1.0,
        anchor_point: float | tuple[float, float] | np.ndarray = (0.0, 0.0),
        origin_point: Direction | str = Direction.CENTER,
        blending_mode: BlendingMode | str = BlendingMode.NORMAL,
    ):
        self.position = Attribute(position, AttributeType.VECTOR2D)
        self.scale = Attribute(scale, AttributeType.VECTOR2D)
        self.rotation = Attribute(rotation, AttributeType.SCALAR)
        self.opacity = Attribute(opacity, AttributeType.SCALAR, range=(0., 1.))
        self.anchor_point = Attribute(anchor_point, AttributeType.VECTOR2D)
        self.origin_point = Direction.from_string(origin_point) if isinstance(origin_point, str) else origin_point
        self.blending_mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode

    @property
    def attributes(self) -> dict[str, Attribute]:
        return {
            "anchor_point": self.anchor_point,
            "position": self.position,
            "scale": self.scale,
            "rotation": self.rotation,
            "opacity": self.opacity,
        }

    @classmethod
    def from_positions(
        cls,
        size: tuple[int, int],
        top: float | None = None,
        bottom: float | None = None,
        left: float | None = None,
        right: float | None = None,
        object_fit: str | None = None
    ) -> "Transform":
        """Allows to create `Transform` by specifying the position based on the edges ``(top, bottom, left, right)``.

        The ``object_fit`` parameter specifies how the object should scale to fit the canvas or container.
        If ``object_fit='contain'``, the object will scale to fit within the canvas while preserving its aspect ratio.
        If ``'cover'``, the object will scale to completely cover the canvas, also preserving its aspect ratio.

        The method calculates the position and origin point of the object based on the supplied arguments
        and returns a new Transform object.

        Args:
            size:
                A tuple of two integers representing the width (``W``) and height (``H``) of the canvas or container.
            top:
                Optional float, distance from the top edge of the layer. Default is ``None``.
            bottom:
                Optional float, distance from the bottom edge of the layer. Default is ``None``.
            left:
                Optional float, distance from the left edge of the layer. Default is ``None``.
            right:
                Optional float, distance from the right edge of the layer. Default is ``None``.
            object_fit:
                Optional string, specifies the scaling behavior. Accepts either ``'contain'`` or ``'cover'``.
                Default is ``None`` (do nothing).

        Returns:
            A new `Transform` object with the specified position, scale, and origin point.
        """
        W, H = size[0], size[1]
        if top is None and bottom is None and left is None and right is None:
            x, y = W / 2, H / 2
            origin_point = Direction.CENTER
            if object_fit is None:
                scale = 1.0
            elif object_fit == "contain":
                scale = min(W, H) / max(W, H)
            elif object_fit == "cover":
                scale = max(W, H) / min(W, H)
            else:
                raise ValueError(f"Invalid object_fit: {object_fit}. contain or cover is expected.")
        else:
            if top is not None and bottom is None and left is None and right is None:
                x, y = W / 2, top
                origin_point = Direction.TOP_CENTER
            elif top is None and bottom is not None and left is None and right is None:
                x, y = W / 2, H - bottom
                origin_point = Direction.BOTTOM_CENTER
            elif top is None and bottom is None and left is not None and right is None:
                x, y = left, H / 2
                origin_point = Direction.CENTER_LEFT
            elif top is None and bottom is None and left is None and right is not None:
                x, y = W - right, H / 2
                origin_point = Direction.CENTER_RIGHT
            elif top is not None and bottom is None and left is not None and right is None:
                x, y = left, top
                origin_point = Direction.TOP_LEFT
            elif top is not None and bottom is None and left is None and right is not None:
                x, y = W - right, top
                origin_point = Direction.TOP_RIGHT
            elif top is None and bottom is not None and left is not None and right is None:
                x, y = left, H - bottom
                origin_point = Direction.BOTTOM_LEFT
            elif top is None and bottom is not None and left is None and right is not None:
                x, y = W - right, H - bottom
                origin_point = Direction.BOTTOM_RIGHT
            else:
                raise ValueError("Invalid pair of arguments")
            scale = 1.0
        return cls(position=(x, y), origin_point=origin_point, scale=scale)

    def get_current_value(self, layer_time: float) -> TransformValue:
        """Retrieves the current transformation attributes for a given time, encapsulated within a ``TransformValue``.

        This includes the anchor point, position, scale, rotation, and opacity, all evaluated at ``layer_time``.
        The transformation attributes are converted to their corresponding 2D vectors or scalar values as appropriate,
        for easy use or manipulation.

        Args:
            layer_time:
                A float representing the time for which you want to get the current value of
                all the transformation attributes.

        Returns:
            ``TransformValue`` that encapsulates the current transformation properties
            (anchor point, position, scale, rotation, opacity, and origin point) for the given ``layer_time``.
        """
        return TransformValue(
            anchor_point=transform_to_2dvector(self.anchor_point(layer_time)),
            position=transform_to_2dvector(self.position(layer_time)),
            scale=transform_to_2dvector(self.scale(layer_time)),
            rotation=transform_to_1dscalar(self.rotation(layer_time)),
            opacity=transform_to_1dscalar(self.opacity(layer_time)),
            origin_point=self.origin_point,
            blending_mode=self.blending_mode,
        )

    def __repr__(self) -> str:
        return f"Transform(ap={self.anchor_point}, pos={self.position}, " \
            f"s={self.scale}, rot={self.rotation}, op={self.opacity}, blend={self.blending_mode})"


def transform_to_1dscalar(x: float | Sequence[float] | np.ndarray) -> float:
    """Transform a scalar, tuple, list or numpy array to a scalar.

    Args:
        x:
            The value to transform.

    Returns:
        A scalar.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, np.ndarray) and x.shape == ():
        return float(x)
    else:
        if 0 < len(x):
            return float(x[0])
        else:
            raise ValueError(f"Invalid value: {x}")


def transform_to_2dvector(
    x: float | Sequence[float] | np.ndarray
) -> tuple[float, float]:
    """Transform a scalar, tuple, list or numpy array to a 2D vector.

    Args:
        x:
            The value to transform.

    Returns:
        A 2D vector.
    """
    if isinstance(x, float):
        return (x, x)
    elif isinstance(x, np.ndarray) and x.shape == ():
        return (float(x), float(x))
    elif len(x) == 1:
        return (float(x[0]), float(x[0]))
    elif len(x) == 2:
        return (float(x[0]), float(x[1]))
    else:
        raise ValueError(f"Invalid value: {x}")


def transform_to_3dvector(
    x: float | Sequence[float] | np.ndarray
) -> tuple[float, float, float]:
    """Transform a scalar, tuple, list or numpy array to a 3D vector.

    Args:
        x:
            The value to transform.

    Returns:
        A 3D vector.
    """
    if isinstance(x, float):
        return (x, x, x)
    elif isinstance(x, np.ndarray) and x.shape == ():
        return (float(x), float(x), float(x))
    elif len(x) == 1:
        y = float(x[0])
        return (y, y, y)
    elif len(x) == 3:
        return (float(x[0]), float(x[1]), float(x[2]))
    else:
        raise ValueError(f"Invalid value: {x}")
