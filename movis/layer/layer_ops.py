from __future__ import annotations
from typing import Hashable

import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..enum import BlendingMode, MatteMode
from ..imgproc import alpha_composite
from .layer import BasicLayer


class AlphaMatte(AttributesMixin):

    def __init__(
            self, mask: BasicLayer, target: BasicLayer,
            opacity: float = 1.0, blending_mode: BlendingMode | str = BlendingMode.NORMAL):
        self.mask = mask
        self.target = target
        self.opacity = Attribute(opacity, value_type=AttributeType.SCALAR, range=(0., 1.))
        self.blending_mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode

    def get_key(self, time: float) -> tuple[Hashable, Hashable, Hashable]:
        attr_key = super().get_key(time)
        mask_key = self.mask.get_key(time) if hasattr(self.mask, 'get_key') else time
        target_key = self.target.get_key(time) if hasattr(self.target, 'get_key') else time
        return (attr_key, mask_key, target_key)

    @property
    def duration(self) -> float:
        return self.mask.duration

    def __call__(self, time: float) -> np.ndarray | None:
        mask_frame = self.mask(time)
        if mask_frame is None:
            return None
        target_frame = self.target(time)
        if target_frame is None:
            return mask_frame
        opacity = float(self.opacity(time))
        return alpha_composite(
            mask_frame, target_frame, opacity=opacity,
            blending_mode=self.blending_mode, matte_mode=MatteMode.ALPHA)


class LuminanceMatte:

    def __init__(self, mask: BasicLayer, target: BasicLayer):
        self.mask = mask
        self.target = target

    def get_key(self, time: float) -> tuple[Hashable, Hashable]:
        mask_key = self.mask.get_key(time) if hasattr(self.mask, 'get_key') else time
        target_key = self.target.get_key(time) if hasattr(self.target, 'get_key') else time
        return (mask_key, target_key)

    @property
    def duration(self) -> float:
        return self.mask.duration

    def __call__(self, time: float) -> np.ndarray | None:
        mask_frame = self.mask(time)
        if mask_frame is None:
            return None
        target_frame = self.target(time)
        if target_frame is None:
            return None
        return alpha_composite(mask_frame, target_frame, matte_mode=MatteMode.LUMINANCE)
