from typing import Hashable, Optional, Union

import numpy as np

from ..enum import BlendingMode
from ..imgproc import alpha_composite
from .layer import Layer


class AlphaMatte:

    def __init__(
            self, source: Layer, target: Layer,
            opacity: float = 1.0, blending_mode: Union[BlendingMode, str] = BlendingMode.NORMAL):
        self.source = source
        self.target = target
        self.opacity = float(opacity)
        self.blending_mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode

    def get_key(self, time: float) -> tuple[Hashable, Hashable]:
        source_key = self.source.get_key(time) if hasattr(self.source, 'get_key') else time
        target_key = self.target.get_key(time) if hasattr(self.target, 'get_key') else time
        return (source_key, target_key)

    @property
    def duration(self) -> float:
        return self.source.duration

    def __call__(self, time: float) -> Optional[np.ndarray]:
        source_frame = self.source(time)
        if source_frame is None:
            return None
        target_frame = self.target(time)
        if target_frame is None:
            return source_frame
        return alpha_composite(
            source_frame, target_frame, opacity=self.opacity,
            blending_mode=self.blending_mode, alpha_matte_mode=True)
