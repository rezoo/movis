from typing import NamedTuple, Optional, Union

import numpy as np

from movis.enum import Direction

from .attribute import Attribute, AttributeType
from .motion import transform_to_1dscalar, transform_to_2dvector


class TransformValue(NamedTuple):
    anchor_point: tuple[float, float] = (0.0, 0.0)
    position: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    opacity: float = 1.0
    origin_point: Direction = Direction.CENTER

    def __post_init__(self):
        if self.opacity < 0.0 or 1.0 < self.opacity:
            raise ValueError("opacity must be in the range [0, 1]")


class Transform:
    def __init__(
        self,
        position: Union[float, tuple[float, float], np.ndarray] = (0.0, 0.0),
        scale: Union[float, tuple[float, float], np.ndarray] = (1.0, 1.0),
        rotation: float = 0.0,
        opacity: float = 1.0,
        anchor_point: Union[float, tuple[float, float], np.ndarray] = (0.0, 0.0),
        origin_point: Union[Direction, str] = Direction.CENTER,
    ):
        self.position = Attribute(position, AttributeType.VECTOR2D)
        self.scale = Attribute(scale, AttributeType.VECTOR2D)
        self.rotation = Attribute(rotation, AttributeType.SCALAR)
        self.opacity = Attribute(opacity, AttributeType.SCALAR, range=(0., 1.))
        self.anchor_point = Attribute(anchor_point, AttributeType.VECTOR2D)
        self.origin_point = Direction.from_string(origin_point) if isinstance(origin_point, str) else origin_point

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
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        left: Optional[float] = None,
        right: Optional[float] = None,
        object_fit: Optional[str] = None
    ) -> "Transform":
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
        return TransformValue(
            anchor_point=transform_to_2dvector(self.anchor_point(layer_time)),
            position=transform_to_2dvector(self.position(layer_time)),
            scale=transform_to_2dvector(self.scale(layer_time)),
            rotation=transform_to_1dscalar(self.rotation(layer_time)),
            opacity=transform_to_1dscalar(self.opacity(layer_time)),
            origin_point=self.origin_point,
        )

    def __repr__(self) -> str:
        return f"Transform(ap={self.anchor_point}, pos={self.position}, " \
            f"s={self.scale}, rot={self.rotation}, op={self.opacity})"
