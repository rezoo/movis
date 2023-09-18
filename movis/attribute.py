from __future__ import annotations

from typing import Callable, Hashable, Sequence

import numpy as np

from movis.enum import AttributeType
from movis.motion import Motion, transform_to_numpy


class Attribute:
    """Attribute class for animating the specified property.

    This class is used for animating layer properties.
    The dimensionality of the values that each Attribute can handle varies
    depending on the type of property.

    This is specified by setting an appropriate value for ``value_type`` using ``AttributeType`` Enum.
    For example, the values will be one-dimensional if ``value_type`` is set to ``AttributeType.SCALAR``.
    If set to ``AttributeType.COLOR``, the values will be three-dimensional.
    Regardless of ``value_type``, the returned type will always be ``numpy.ndarray``.

    .. note::
        Even if it's scalar, the returned value will be an array like ``np.array([value])``.

    Args:
        init_value:
            The initial value to use. It is used when no animation is set,
            or when an animation with zero keyframes is set.
        value_type:
            Specifies the type of value that the Attribute will handle.
            For more details, see the docs of ``AttributeType``.
        range:
            Defines the upper and lower limits of the possible values.
            If set, the returned array's values are guaranteed to be within this range;
            values that exceed this range are simply clipped.
        motion:
            The instance of the motion to use when adding keyframe animations.
            It can be specified in the constructor or activated later using ``enable_motion()``.
        functions:
            User-defined functions for adding animations, separate from keyframe animations.
            It can be specified in the constructor or added later using ``add_function()``.
    """
    def __init__(
        self,
        init_value: float | tuple[float, ...] | np.ndarray,
        value_type: AttributeType,
        range: tuple[float, float] | None = None,
        motion: Motion | None = None,
        functions: Sequence[Callable[[np.ndarray, float], np.ndarray]] | None = None,
    ) -> None:
        np_value = transform_to_numpy(init_value, value_type)
        clipped_value = np.clip(np_value, range[0], range[1]) if range is not None else np_value
        self._init_value: np.ndarray = clipped_value
        self._value_type = value_type
        self._range = range
        self._motion = motion
        self._functions = [] if functions is None else list(functions)

    def __call__(self, layer_time: float) -> np.ndarray:
        if self._motion is None and len(self._functions) == 0:
            return transform_to_numpy(self._init_value, self._value_type)
        else:
            value = self._init_value
            if self._motion is not None:
                value = transform_to_numpy(
                    self._motion(value, layer_time), self._value_type)
            if 0 < len(self._functions):
                for func in self._functions:
                    value = transform_to_numpy(
                        func(value, layer_time), self._value_type)
            if self._range is not None:
                return np.clip(value, self._range[0], self._range[1])
            else:
                return value

    def get_values(self, layer_times: np.ndarray) -> np.ndarray:
        """Returns an array of values for the specified layer times.

        Args:
            layer_times: An array of times for which to get the values.

        Returns:
            An array of values for the specified layer times.
        """
        assert layer_times.ndim == 1
        N, C = len(layer_times), len(self._init_value)
        if self._motion is None and len(self._functions) == 0:
            return np.broadcast_to(self._init_value.reshape(1, C), (N, C))
        else:
            values = [self(t) for t in layer_times]
            return np.concatenate(values).reshape(N, C)

    def enable_motion(self) -> Motion:
        """Enable `Motion` object to animate the attribute."""
        if self._motion is None:
            motion = Motion(init_value=self._init_value, value_type=self._value_type)
            self._motion = motion
        return self._motion

    def disable_motion(self) -> None:
        """Remove `Motion` object."""
        self._motion = None

    @property
    def init_value(self) -> np.ndarray:
        """The initial value of the attribute."""
        return self._init_value

    @init_value.setter
    def init_value(self, value: float | tuple[float, ...] | np.ndarray) -> None:
        value = transform_to_numpy(value, self._value_type)
        self._init_value = value
        if self._motion is not None:
            self._motion.init_value = value

    def set(self, init_value: float | Sequence[float] | np.ndarray) -> None:
        """Set the initial value of the attribute.

        .. note::
            This method is equivalent to ``init_value = value``.

        Args:
            init_value: The value to set.

        Examples:
            >>> import movis as mv
            >>> layer = mv.layer.Rectangle(size=(100, 100), color=(255, 0, 0))
            >>> layer.size.set((200, 200))
            >>> layer.color.set((0, 255, 0))
        """
        value = transform_to_numpy(init_value, self._value_type)
        self._init_value = value
        if self._motion is not None:
            self._motion.init_value = value

    @property
    def value_type(self) -> AttributeType:
        """The type of value that the attribute handles."""
        return self._value_type

    @property
    def range(self) -> tuple[float, float] | None:
        """The upper and lower limits of the possible values."""
        return self._range

    @range.setter
    def range(self, range: tuple[float, float] | None) -> None:
        self._range = range

    @property
    def motion(self) -> Motion | None:
        """The instance of the motion to use when adding keyframe animations."""
        return self._motion

    @property
    def functions(self) -> list[Callable[[np.ndarray, float], np.ndarray]]:
        """User-defined functions for adding animations, separate from keyframe animations."""
        return self._functions

    def add_function(
        self, function: Callable[[np.ndarray, float], np.ndarray]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """Add a user-defined function for adding animations, separate from keyframe animations.

        Args:
            function: A function that takes two arguments, `value` and `time` and returns an array.

        Returns:
            The function that was added.
        """
        if not callable(function):
            raise ValueError(f"Invalid function: {function}")
        self._functions.append(function)
        return function

    def pop_function(self, index: int) -> Callable[[np.ndarray, float], np.ndarray]:
        """Remove a user-defined function of the specified index."""
        return self._functions.pop(index)

    def clear_functions(self) -> None:
        """Remove all user-defined functions."""
        self._functions.clear()

    def __getitem__(self, index: int) -> Callable[[np.ndarray, float], np.ndarray]:
        return self._functions[index]

    def __repr__(self) -> str:
        if self._motion is None:
            return f"{self._init_value}"
        else:
            return f"Attribute(value_type={self._value_type})"


class AttributesMixin:
    """A mix-in class with a collection of methods for implementing layers or effects using attributes.

    When using attributes, the content of layers or effects changes based on those attributes.
    Specifically, these attributes influence the generation of cache keys for the composition.

    This mixin class adds a feature that, when the ``get_key()`` method is called,
    extracts all attributes from the instance and converts them into a format that can be used for the cache key.
    """

    @property
    def attributes(self) -> dict[str, Attribute]:
        """A dictionary of attributes that are used to generate cache keys."""
        return {key: attr for key, attr in vars(self).items() if isinstance(attr, Attribute)}

    def get_key(self, time: float) -> tuple[Hashable, ...]:
        """Returns a tuple of hashable values that represent the state of the instance at a given time."""
        return tuple([transform_to_hashable(attr(time)) for attr in self.attributes.values()])


def transform_to_hashable(
    x: float | Sequence[float] | np.ndarray
) -> float | tuple[float, ...]:
    """Transform a scalar, tuple, list or numpy array to a hashable object used for caching.

    Args:
        x: The value to transform.

    Returns:
        A hashable object.
    """
    if isinstance(x, (int, float)):
        return float(x)
    elif len(x) == 1:
        return float(x[0])
    else:
        return tuple([float(v) for v in x])
