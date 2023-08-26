from typing import Hashable, Optional, Protocol

import numpy as np


class Layer(Protocol):
    def __call__(self, time: float) -> Optional[np.ndarray]:
        raise NotImplementedError

    @property
    def duration(self):
        raise NotImplementedError

    def get_key(self, time: float) -> Hashable:
        return time

    @property
    def use_global_time(self) -> bool:
        return False
