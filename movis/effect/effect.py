from typing import Hashable, Protocol

import numpy as np


class Effect(Protocol):

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        return time
