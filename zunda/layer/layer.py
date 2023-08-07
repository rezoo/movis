from typing import Hashable, Optional, Protocol

import numpy as np


class Layer(Protocol):
    @property
    def duration(self):
        raise NotImplementedError

    def __call__(self, time: float) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        return time
