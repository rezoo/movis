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
    EASE_IN_CUBIC = 4
    EASE_OUT_CUBIC = 6
    EASE_IN_EXPO = 7
    EASE_OUT_EXPO = 8

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
    "ease_in_cubic": MotionType.EASE_IN_CUBIC,
    "ease_out_cubic": MotionType.EASE_OUT_CUBIC,
    "ease_in_expo": MotionType.EASE_IN_EXPO,
    "ease_out_expo": MotionType.EASE_OUT_EXPO,
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
