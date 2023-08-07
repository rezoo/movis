from typing import Hashable, Protocol

import numpy as np

from zunda.attribute import Attribute


class Effect(Protocol):

    def __call__(self, time: float, prev_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        return time

    @property
    def attributes(self) -> dict[str, Attribute]:
        return dict()
