from typing import NamedTuple, Union

import numpy as np

from .attribute import Attribute, AttributeType
from .motion import transform_to_1dscalar, transform_to_2dvector


class TransformValue(NamedTuple):
    anchor_point: tuple[float, float] = (0.0, 0.0)
    position: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    opacity: float = 1.0

    def __post_init__(self):
        if self.opacity < 0.0 or 1.0 < self.opacity:
            raise ValueError("opacity must be in the range [0, 1]")


class Transform:
    def __init__(
        self,
        anchor_point: Union[float, tuple[float, float], np.ndarray] = (0.0, 0.0),
        position: Union[float, tuple[float, float], np.ndarray] = (0.0, 0.0),
        scale: Union[float, tuple[float, float], np.ndarray] = (1.0, 1.0),
        rotation: float = 0.0,
        opacity: float = 1.0
    ):
        self.anchor_point = Attribute(anchor_point, AttributeType.VECTOR2D)
        self.position = Attribute(position, AttributeType.VECTOR2D)
        self.scale = Attribute(scale, AttributeType.VECTOR2D)
        self.rotation = Attribute(rotation, AttributeType.SCALAR)
        self.opacity = Attribute(opacity, AttributeType.SCALAR, range=(0., 1.))

    @property
    def attributes(self) -> dict[str, Attribute]:
        return {
            "anchor_point": self.anchor_point,
            "position": self.position,
            "scale": self.scale,
            "opacity": self.opacity,
        }

    def get_current_value(self, layer_time: float) -> TransformValue:
        return TransformValue(
            anchor_point=transform_to_2dvector(self.anchor_point(layer_time)),
            position=transform_to_2dvector(self.position(layer_time)),
            scale=transform_to_2dvector(self.scale(layer_time)),
            rotation=transform_to_1dscalar(self.rotation(layer_time)),
            opacity=transform_to_1dscalar(self.opacity(layer_time)),
        )

    def __repr__(self) -> str:
        return f"Transform(ap={self.anchor_point}, pos={self.position}, " \
            f"s={self.scale}, rot={self.rotation}, op={self.opacity})"
