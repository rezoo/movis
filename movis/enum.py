from enum import Enum


class CacheType(Enum):
    COMPOSITION = 0
    LAYER = 1


class AttributeType(Enum):
    ANY = -1
    SCALAR = 0
    VECTOR2D = 1
    VECTOR3D = 2
    ANGLE = 3
    COLOR = 4

    @staticmethod
    def from_string(s: str) -> "AttributeType":
        if s == 'scalar':
            return AttributeType.SCALAR
        elif s == 'vector2d':
            return AttributeType.VECTOR2D
        elif s == 'vector3d':
            return AttributeType.VECTOR3D
        elif s == 'angle':
            return AttributeType.ANGLE
        else:
            raise ValueError(f"Unknown attribute type: {s}")


class MotionType(Enum):
    LINEAR = 0
    EASE_IN = 1
    EASE_OUT = 2
    EASE_IN_OUT = 3
    EASE_IN_MEDIUM = 4
    EASE_OUT_MEDIUM = 5
    EASE_IN_OUT_MEDIUM = 6
    EASE_IN_EXPO = 7
    EASE_OUT_EXPO = 8
    EASE_IN_OUT_EXPO = 9

    @staticmethod
    def from_string(s: str) -> "MotionType":
        if s in STRING_TO_MOTION_TYPE:
            return STRING_TO_MOTION_TYPE[s]
        else:
            raise ValueError(f"Unknown motion type: {s}")


STRING_TO_MOTION_TYPE = {
    "linear": MotionType.LINEAR,
    "ease_in": MotionType.EASE_IN,
    "ease_out": MotionType.EASE_OUT,
    "ease_in_out": MotionType.EASE_IN_OUT,
    "ease_in_medium": MotionType.EASE_IN_MEDIUM,
    "ease_out_medium": MotionType.EASE_OUT_MEDIUM,
    "ease_in_out_medium": MotionType.EASE_IN_OUT_MEDIUM,
    "ease_in_expo": MotionType.EASE_IN_EXPO,
    "ease_out_expo": MotionType.EASE_OUT_EXPO,
    "ease_in_out_expo": MotionType.EASE_IN_OUT_EXPO,
}


class BlendingMode(Enum):
    NORMAL = 0
    MULTIPLY = 1
    SCREEN = 2
    OVERLAY = 3
    HARD_LIGHT = 4
    SOFT_LIGHT = 5

    @staticmethod
    def from_string(s: str) -> "BlendingMode":
        if s in STRING_TO_BLENDING_MODE:
            return STRING_TO_BLENDING_MODE[s]
        else:
            raise ValueError(f"Unknown blending mode: {s}")


STRING_TO_BLENDING_MODE = {
    "normal": BlendingMode.NORMAL,
    "multiply": BlendingMode.MULTIPLY,
    "screen": BlendingMode.SCREEN,
    "overlay": BlendingMode.OVERLAY,
    "hard_light": BlendingMode.HARD_LIGHT,
    "soft_light": BlendingMode.SOFT_LIGHT,
}


class Direction(Enum):
    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 2
    BOTTOM_RIGHT = 3
    CENTER_LEFT = 4
    CENTER = 5
    CENTER_RIGHT = 6
    TOP_LEFT = 7
    TOP_CENTER = 8
    TOP_RIGHT = 9

    @staticmethod
    def from_string(s: str) -> "Direction":
        if s in STRING_TO_DIRECTION:
            return STRING_TO_DIRECTION[s]
        else:
            raise ValueError(f"Unknown origin point: {s}")

    @staticmethod
    def to_vector(d: "Direction", size: tuple[float, float]) -> tuple[float, float]:
        if d == Direction.BOTTOM_LEFT:
            return (0, size[1])
        elif d == Direction.BOTTOM_CENTER:
            return (size[0] / 2, size[1])
        elif d == Direction.BOTTOM_RIGHT:
            return (size[0], size[1])
        elif d == Direction.CENTER_LEFT:
            return (0, size[1] / 2)
        elif d == Direction.CENTER:
            return (size[0] / 2, size[1] / 2)
        elif d == Direction.CENTER_RIGHT:
            return (size[0], size[1] / 2)
        elif d == Direction.TOP_LEFT:
            return (0, 0)
        elif d == Direction.TOP_CENTER:
            return (size[0] / 2, 0)
        elif d == Direction.TOP_RIGHT:
            return (size[0], 0)
        else:
            raise ValueError(f"Unknown direction: {d}")


STRING_TO_DIRECTION = {
    "bottom_left": Direction.BOTTOM_LEFT,
    "bottom_center": Direction.BOTTOM_CENTER,
    "bottom_right": Direction.BOTTOM_RIGHT,
    "center_left": Direction.CENTER_LEFT,
    "center": Direction.CENTER,
    "center_right": Direction.CENTER_RIGHT,
    "top_left": Direction.TOP_LEFT,
    "top_center": Direction.TOP_CENTER,
    "top_right": Direction.TOP_RIGHT,
}


class TextAlignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2

    @staticmethod
    def from_string(s: str) -> "TextAlignment":
        if s in STRING_TO_TEXT_ALIGNMENT:
            return STRING_TO_TEXT_ALIGNMENT[s]
        else:
            raise ValueError(f"Unknown text alignment: {s}")


STRING_TO_TEXT_ALIGNMENT = {
    "left": TextAlignment.LEFT,
    "center": TextAlignment.CENTER,
    "right": TextAlignment.RIGHT,
}
