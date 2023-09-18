from __future__ import annotations

from typing import Hashable

import numpy as np

from ..attribute import Attribute, AttributesMixin, AttributeType
from ..enum import BlendingMode, MatteMode
from ..imgproc import alpha_composite
from .protocol import BasicLayer


class AlphaMatte(AttributesMixin):
    """A layer that applies alpha matte to the target layer using the mask layer.

    Alpha Matte is a algorihtm that overlays the target layer on the mask layer without changing the alpha channel.
    The mask layer and the target layer should have the same size and the same duration.
    Using the `Composition` layer is preferred if users want to align them.

    Args:
        mask:
            the base mask layer used for alpha matte. ``mask`` must comply with the ``BasicLayer`` protocol.
        target:
            the target layer to which alpha matte is applied. ``target`` must comply with the ``BasicLayer`` protocol.
        opacity:
            the opacity of the target layer. Defaults to ``1.0``.
        blending_mode:
            the blending mode of the target layer. Defaults to ``BlendingMode.NORMAL``.

    Animatable Attributes:
        ``opacity``
    """

    def __init__(
            self, mask: BasicLayer, target: BasicLayer,
            opacity: float = 1.0, blending_mode: BlendingMode | str = BlendingMode.NORMAL):
        self.mask = mask
        self.target = target
        self.opacity = Attribute(opacity, value_type=AttributeType.SCALAR, range=(0., 1.))
        self.blending_mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode

    def get_key(self, time: float) -> tuple[Hashable, Hashable, Hashable]:
        """Get the state for the given time."""
        attr_key = super().get_key(time)
        mask_key = self.mask.get_key(time) if hasattr(self.mask, 'get_key') else time
        target_key = self.target.get_key(time) if hasattr(self.target, 'get_key') else time
        return (attr_key, mask_key, target_key)

    @property
    def duration(self) -> float:
        """The duration of the layer."""
        return self.mask.duration

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0 or self.duration <= time:
            return None
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
    """A layer that replaces the alpha channel of the target layer with the luminance of the mask layer.

    .. note::
        The mask layer and the target layer should have the same size and the same duration.
        Using the `Composition` layer is preferred if users want to align them.

    Args:
        mask:
            the base mask layer used for luminance matte. ``mask`` must comply with the ``BasicLayer`` protocol.
        target:
            the target layer to which luminance matte is applied.
            ``target`` must comply with the ``BasicLayer`` protocol.
    """

    def __init__(self, mask: BasicLayer, target: BasicLayer):
        self.mask = mask
        self.target = target

    def get_key(self, time: float) -> tuple[Hashable, Hashable]:
        """Get the state for the given time."""
        mask_key = self.mask.get_key(time) if hasattr(self.mask, 'get_key') else time
        target_key = self.target.get_key(time) if hasattr(self.target, 'get_key') else time
        return (mask_key, target_key)

    @property
    def duration(self) -> float:
        """The duration of the layer."""
        return self.mask.duration

    def __call__(self, time: float) -> np.ndarray | None:
        if time < 0 or self.duration <= time:
            return None
        mask_frame = self.mask(time)
        if mask_frame is None:
            return None
        target_frame = self.target(time)
        if target_frame is None:
            return None
        return alpha_composite(mask_frame, target_frame, matte_mode=MatteMode.LUMINANCE)
