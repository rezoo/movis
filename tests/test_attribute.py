import numpy as np
import pytest

from movis.attribute import Attribute, AttributeType

attribute_params = [
    (0.5, AttributeType.SCALAR),
    (1.0, AttributeType.ANGLE),
    (0.3, AttributeType.COLOR),
    (0.2, AttributeType.VECTOR2D),
    (0.9, AttributeType.VECTOR3D),
]


@pytest.mark.parametrize("value, value_type", attribute_params)
def test_attribute_init(value: float, value_type: AttributeType):
    attr = Attribute(value, value_type)
    assert isinstance(attr.init_value, np.ndarray)
    if value_type == AttributeType.SCALAR or value_type == AttributeType.ANGLE:
        assert attr.init_value.shape == (1,)
        val = attr(0.0)
        assert val.shape == (1,)
    elif value_type == AttributeType.VECTOR2D:
        assert attr.init_value.shape == (2,)
        val = attr(0.0)
        assert val.shape == (2,)
    elif value_type == AttributeType.VECTOR3D or value_type == AttributeType.COLOR:
        assert attr.init_value.shape == (3,)
        val = attr(0.0)
        assert val.shape == (3,)
    assert isinstance(val, np.ndarray)
    assert val.dtype == np.float64


@pytest.mark.parametrize("value, value_type", attribute_params)
def test_attribute_init_range(value: float, value_type: AttributeType):
    attr = Attribute(value, value_type, range=(0.0, 0.1))
    assert isinstance(attr.init_value, np.ndarray)
    if value_type == AttributeType.SCALAR or value_type == AttributeType.ANGLE:
        assert attr.init_value.shape == (1,)
        val = attr(0.0)
        assert val.shape == (1,)
    elif value_type == AttributeType.VECTOR2D:
        assert attr.init_value.shape == (2,)
        val = attr(0.0)
        assert val.shape == (2,)
    elif value_type == AttributeType.VECTOR3D or value_type == AttributeType.COLOR:
        assert attr.init_value.shape == (3,)
        val = attr(0.0)
        assert val.shape == (3,)
    assert np.all(val == 0.1)
