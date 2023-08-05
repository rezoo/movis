from typing import Hashable, Protocol

import numpy as np

from zunda.attribute import Attribute


class Effect(Protocol):

    def get_key(self, time: float) -> Hashable:
        raise NotImplementedError

    def __call__(self, time: float, prev_value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def attributes(self) -> dict[str, Attribute]:
        raise NotImplementedError


class BasicEffectMixin:

    def get_key(self, time: float) -> Hashable:
        # Assume that the effect is not time-dependent
        return 0

    @property
    def attributes(self) -> dict[str, Attribute]:
        # Assume that the effect has no attributes
        return {}
