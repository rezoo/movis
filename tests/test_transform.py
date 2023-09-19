import numpy as np
import pytest

from movis.attribute import Attribute
from movis.transform import (Direction, Transform, TransformValue,
                             transform_to_1dscalar, transform_to_2dvector,
                             transform_to_3dvector)


def test_anchor_point():
    transform = Transform(anchor_point=(0.5, 0.75))
    assert np.all(transform.anchor_point.init_value == (0.5, 0.75))
    assert np.all(transform.anchor_point(0.0) == (0.5, 0.75))
    assert np.all(transform.anchor_point(0.5) == (0.5, 0.75))
    assert np.all(transform.anchor_point(1.0) == (0.5, 0.75))


def test_position():
    transform = Transform(position=(0.5, 1.0))
    assert np.all(transform.position.init_value == (0.5, 1.0))
    assert np.all(transform.position(0.0) == (0.5, 1.0))
    assert np.all(transform.position(0.5) == (0.5, 1.0))
    assert np.all(transform.position(1.0) == (0.5, 1.0))


def test_scale():
    transform = Transform(scale=0.5)
    assert np.all(transform.scale.init_value == (0.5, 0.5))
    assert np.all(transform.scale(0.0) == (0.5, 0.5))
    assert np.all(transform.scale(0.5) == (0.5, 0.5))
    assert np.all(transform.scale(1.0) == (0.5, 0.5))

    transform = Transform(scale=(0.5, 1.0))
    assert np.all(transform.scale.init_value == (0.5, 1.0))
    assert np.all(transform.scale(0.0) == (0.5, 1.0))
    assert np.all(transform.scale(0.5) == (0.5, 1.0))
    assert np.all(transform.scale(1.0) == (0.5, 1.0))


def test_rotation():
    transform = Transform(rotation=45.0)
    assert np.all(transform.rotation.init_value == 45.0)
    assert np.all(transform.rotation(0.0) == 45.0)
    assert np.all(transform.rotation(0.5) == 45.0)
    assert np.all(transform.rotation(1.0) == 45.0)


def test_opacity():
    transform = Transform(opacity=0.5)
    assert np.all(transform.opacity.init_value == 0.5)
    assert np.all(transform.opacity(0.0) == 0.5)
    assert np.all(transform.opacity(0.5) == 0.5)
    assert np.all(transform.opacity(1.0) == 0.5)


def test_attributes():
    transform = Transform()
    assert isinstance(transform.attributes, dict)
    assert len(transform.attributes) == 5
    assert "anchor_point" in transform.attributes
    assert "position" in transform.attributes
    assert "scale" in transform.attributes
    assert "rotation" in transform.attributes
    assert "opacity" in transform.attributes


def test_get_current_value():
    transform = Transform()
    assert isinstance(transform.get_current_value(0.0), TransformValue)
    assert isinstance(transform.get_current_value(0.5), TransformValue)
    assert isinstance(transform.get_current_value(1.0), TransformValue)

    v = transform.get_current_value(0.0)
    assert isinstance(v.anchor_point, tuple)
    assert isinstance(v.position, tuple)
    assert isinstance(v.scale, tuple)
    assert isinstance(v.rotation, float)
    assert isinstance(v.opacity, float)
    assert isinstance(v.origin_point, Direction)


def test_get_attributes():
    transform = Transform()
    attrs = transform.attributes
    assert isinstance(attrs, dict)
    assert len(attrs) == 5

    for attr in attrs.values():
        assert isinstance(attr, Attribute)


def test_from_positions():
    transform = Transform.from_positions((100, 100))
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([50.0, 50.0]))

    transform = Transform.from_positions((100, 100), top=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([50.0, 0.0]))
    assert transform.origin_point == Direction.TOP_CENTER

    transform = Transform.from_positions((100, 100), bottom=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([50.0, 100.0]))
    assert transform.origin_point == Direction.BOTTOM_CENTER

    transform = Transform.from_positions((100, 100), left=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([0.0, 50.0]))
    assert transform.origin_point == Direction.CENTER_LEFT

    transform = Transform.from_positions((100, 100), right=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([100.0, 50.0]))
    assert transform.origin_point == Direction.CENTER_RIGHT

    transform = Transform.from_positions((100, 100), top=0.0, left=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([0.0, 0.0]))
    assert transform.origin_point == Direction.TOP_LEFT

    transform = Transform.from_positions((100, 100), top=0.0, right=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([100.0, 0.0]))
    assert transform.origin_point == Direction.TOP_RIGHT

    transform = Transform.from_positions((100, 100), bottom=0.0, left=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([0.0, 100.0]))
    assert transform.origin_point == Direction.BOTTOM_LEFT

    transform = Transform.from_positions((100, 100), bottom=0.0, right=0.0)
    assert isinstance(transform, Transform)
    assert np.all(transform.position.init_value == np.array([100.0, 100.0]))
    assert transform.origin_point == Direction.BOTTOM_RIGHT


def test_transform_value_defaults():
    transform = TransformValue()
    assert transform.anchor_point == (0.0, 0.0)
    assert transform.position == (0.0, 0.0)
    assert transform.scale == (1.0, 1.0)
    assert transform.rotation == 0.0
    assert transform.opacity == 1.0
    assert transform.origin_point == Direction.CENTER


def test_transform_value_custom_values():
    transform = TransformValue(anchor_point=(1.0, 2.0), position=(3.0, 4.0), scale=(2.0, 2.0),
                               rotation=45.0, opacity=0.5, origin_point=Direction.TOP_LEFT)
    assert transform.anchor_point == (1.0, 2.0)
    assert transform.position == (3.0, 4.0)
    assert transform.scale == (2.0, 2.0)
    assert transform.rotation == 45.0
    assert transform.opacity == 0.5
    assert transform.origin_point == Direction.TOP_LEFT


def test_transform_to_1dscalar():
    assert transform_to_1dscalar(3.0) == 3.0
    assert transform_to_1dscalar([3.0]) == 3.0
    assert transform_to_1dscalar((3.0,)) == 3.0
    assert transform_to_1dscalar(np.array(3.0)) == 3.0
    assert transform_to_1dscalar(np.array([3.0])) == 3.0

    with pytest.raises(ValueError):
        transform_to_1dscalar([])


def test_transform_to_2dvector():
    assert transform_to_2dvector(3.0) == (3.0, 3.0)
    assert transform_to_2dvector([3.0]) == (3.0, 3.0)
    assert transform_to_2dvector((3.0,)) == (3.0, 3.0)
    assert transform_to_2dvector([3.0, 4.0]) == (3.0, 4.0)
    assert transform_to_2dvector(np.array(3.0)) == (3.0, 3.0)
    assert transform_to_2dvector(np.array([3.0])) == (3.0, 3.0)

    with pytest.raises(ValueError):
        transform_to_2dvector([3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        transform_to_2dvector([])


def test_transform_to_3dvector():
    assert transform_to_3dvector(3.0) == (3.0, 3.0, 3.0)
    assert transform_to_3dvector([3.0]) == (3.0, 3.0, 3.0)
    assert transform_to_3dvector((3.0,)) == (3.0, 3.0, 3.0)
    assert transform_to_3dvector([3.0, 4.0, 5.0]) == (3.0, 4.0, 5.0)
    assert transform_to_3dvector(np.array(3.0)) == (3.0, 3.0, 3.0)
    assert transform_to_3dvector(np.array([3.0])) == (3.0, 3.0, 3.0)

    with pytest.raises(ValueError):
        transform_to_3dvector([3.0, 4.0])
    with pytest.raises(ValueError):
        transform_to_3dvector([3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        transform_to_3dvector([])
