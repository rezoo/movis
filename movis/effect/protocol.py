from typing import Hashable, Protocol

import numpy as np


class Effect(Protocol):
    """The protocol that defimes the minimal interface for an effect."""

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        """The minimum required method to implement an effect. All effects must implement it.

        Specifically, given a previous image described as ``prev_image`` and a time,
        it returns a ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as
        ``numpy.uint8`` given a time.

        Args:
            prev_time:
                A ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as ``numpy.uint8``.
            time:
                A scalar variable representing time.

        Returns:
            ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as ``numpy.uint8``.
        """
        raise NotImplementedError


class BasicEffect(Effect):
    """The protocol that defines the basic interface for an effect with some methods."""

    def get_key(self, time: float) -> Hashable:
        """An optional but desirable method for any layer implementation.

        This method returns a hashable value representing the 'state' of the effect at a given time.
        If the keys are the same and given images are identical,
        the array returned by this layer must also be identical.
        It is used for caching compositions, and in the case of videos where static frames persist,
        Movis will use the cache to accelerate video rendering.

        If not implemented, Movis assumes that the layer is independent at each time frame,
        `i.e.`, it will not use cache-based rendering.

        Returns:
            A hashable key that represents the state of the layer at the given time.
        """
        return time
