import numpy as np

from movis.transform import Transform, TransformValue


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
