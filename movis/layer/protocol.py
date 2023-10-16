from __future__ import annotations

from typing import Hashable, Protocol

import numpy as np

AUDIO_SAMPLING_RATE = 44100
AUDIO_BLOCK_SIZE = 1024


class Layer(Protocol):
    """The protocol that defimes the minimal interface for a layer."""

    def __call__(self, time: float) -> np.ndarray | None:
        """The minimum required method to implement a layer. All layers must implement it.

        Specifically, this method returns a ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as
        ``numpy.uint8``, or ``None``, given a time.
        When a ``numpy.ndarray`` is returned, Movis considers this array as an image and uses it as one of the layers
        for rendering the video. If ``None`` is returned, Movis does not render its layer.

        Args:
            time: A scalar variable representing time.

        Returns:
            ``None`` if nothing is to be rendered, otherwise ``numpy.ndarray``.
        """
        raise NotImplementedError


class BasicLayer(Protocol):
    """The protocol that defines the basic interface for a layer with some optional properties."""

    def __call__(self, time: float) -> np.ndarray | None:
        """The minimum required method to implement a layer. All layers must implement it.

        Specifically, this method returns a ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as
        ``numpy.uint8``, or ``None``, given a time.
        When a ``numpy.ndarray`` is returned, Movis considers this array as an image and uses it as one of the layers
        for rendering the video. If ``None`` is returned, Movis does not render its layer.

        Args:
            time: A scalar variable representing time.

        Returns:
            ``None`` if nothing is to be rendered, otherwise ``numpy.ndarray``.
        """
        raise NotImplementedError

    @property
    def duration(self) -> float:
        """An optional but desirable property for any layer implementation.

        This property should return the duration for which the layer will persist.
        If not implemented, it is assumed that the layer has an indefinitely large duration.

        Returns:
            The duration for which the layer will persist.
        """
        return 1e6

    def get_key(self, time: float) -> Hashable:
        """An optional but desirable method for any layer implementation.

        This method returns a hashable value representing the 'state' of the layer at a given time.
        If the keys are the same, the array returned by this layer must also be identical.
        It is used for caching compositions, and in the case of videos where static frames persist,
        Movis will use the cache to accelerate video rendering.

        If not implemented, Movis assumes that the layer is independent at each time frame,
        `i.e.`, it will not use cache-based rendering.

        Returns:
            A hashable key that represents the state of the layer at the given time.
        """
        return time


class AudioLayer(Protocol):

    def __call__(self, time: float) -> np.ndarray | None:
        """The minimum required method to implement a layer. All layers must implement it.

        Specifically, this method returns a ``numpy.ndarray`` of shape ``(H, W, 4)`` with RGBA order and dtype as
        ``numpy.uint8``, or ``None``, given a time.
        When a ``numpy.ndarray`` is returned, Movis considers this array as an image and uses it as one of the layers
        for rendering the video. If ``None`` is returned, Movis does not render its layer.

        Args:
            time: A scalar variable representing time.

        Returns:
            ``None`` if nothing is to be rendered, otherwise ``numpy.ndarray``.
        """
        raise NotImplementedError

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        """An optional method for implementing an audio layer.

        This method returns an audio clip of the layer between the given start and end times.
        The returned audio clip should be a two-dimensional ``numpy.ndarray`` with a shape of ``(2,T)``,
        where ``T`` is the number of samples in the audio clip and ``2`` is the number of channels.
        The sample rate of the audio clip should be ``AUDIO_SAMPLING_RATE`` (= 44100).

        If not implemented, Movis assumes that the layer does not have any audio.

        Args:
            start_time: The start time of the audio clip.
            end_time: The end time of the audio clip.

        Returns:
            A two-dimensional ``numpy.ndarray`` with a shape of ``(2,T)`` and ``sample_rate``
            is an integer representing the sample rate of the audio clip.
        """
        raise NotImplementedError
