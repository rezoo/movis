import numpy as np
import pytest

from movis.attribute import AttributeType
from movis.motion import Motion
from movis import Easing


def test_motion_call():
    motion = Motion(init_value=0.5)
    assert np.all(motion(None, 0.0) == np.array([0.5]))
    assert np.all(motion(None, 1.0) == np.array([0.5]))
    motion = Motion(init_value=0.5, value_type=AttributeType.SCALAR)
    assert np.all(motion(None, 0.0) == np.array([0.5]))
    assert np.all(motion(None, 1.0) == np.array([0.5]))
    motion = Motion(init_value=0.5, value_type=AttributeType.ANGLE)
    assert np.all(motion(None, 0.0) == np.array([0.5]))
    assert np.all(motion(None, 1.0) == np.array([0.5]))
    motion = Motion(init_value=0.5, value_type=AttributeType.VECTOR2D)
    assert np.all(motion(None, 0.0) == np.array([0.5, 0.5]))
    motion = Motion(init_value=(0.5, 0.5), value_type=AttributeType.VECTOR2D)
    assert np.all(motion(None, 1.0) == np.array([0.5, 0.5]))
    motion = Motion(init_value=0.5, value_type=AttributeType.VECTOR3D)
    assert np.all(motion(None, 0.0) == np.array([0.5, 0.5, 0.5]))
    motion = Motion(init_value=(0.5, 0.5, 0.5), value_type=AttributeType.VECTOR3D)
    assert np.all(motion(None, 0.0) == np.array([0.5, 0.5, 0.5]))
    assert np.all(motion(None, 1.0) == np.array([0.5, 0.5, 0.5]))
    motion = Motion(init_value=0.5, value_type=AttributeType.COLOR)
    assert np.all(motion(None, 0.0) == np.array([0.5, 0.5, 0.5]))
    motion = Motion(init_value=(0.5, 0.5, 0.5), value_type=AttributeType.COLOR)
    assert np.all(motion(None, 0.0) == np.array([0.5, 0.5, 0.5]))
    assert np.all(motion(None, 1.0) == np.array([0.5, 0.5, 0.5]))


motion_append_params = [
    (AttributeType.SCALAR, 1),
    (AttributeType.ANGLE, 1),
    (AttributeType.VECTOR2D, 2),
    (AttributeType.VECTOR3D, 3),
    (AttributeType.COLOR, 3),
]


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_append(value_type, n_dim):
    motion = Motion(init_value=0.0, value_type=value_type)
    motion.append(0.0, 0.0)
    motion.append(1.0, 1.0)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_append_with_easing(value_type, n_dim):
    motion = Motion(init_value=0.0, value_type=value_type)
    motion.append(0.0, 0.0, easing=Easing.LINEAR)
    motion.append(1.0, 1.0, easing=Easing.LINEAR)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))

    motion = Motion(init_value=0.0, value_type=value_type)
    motion.append(0.0, 0.0, easing='linear')
    motion.append(1.0, 1.0, easing='linear')
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))

    def easing_func(t):
        return t

    motion = Motion(init_value=0.0, value_type=value_type)
    motion.append(0.0, 0.0, easing=easing_func)
    motion.append(1.0, 1.0, easing=easing_func)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_append_full(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.append(0.0, [0.0] * n_dim)
    motion.append(1.0, [1.0] * n_dim)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_append_3(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.append(0.0, 0.0)
    motion.append(1.0, 1.0)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))
    motion.append(0.5, 0.0)
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.75) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_extend(value_type, n_dim):
    motion = Motion(init_value=0.0, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0])
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_extend_full(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(
        keyframes=[0.0, 1.0],
        values=[[0.0] * n_dim, [1.0] * n_dim])
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_extend_3(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0])

    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
    assert np.all(motion(None, 2.0) == np.array([1.0] * n_dim))

    motion.extend(keyframes=[0.5], values=[0.0])
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.75) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_clear(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 0.5, 1.0], values=[1.0, 1.0, 2.0])
    motion.clear()
    assert np.all(motion(None, -1.0) == np.array([0.0] * n_dim))


@pytest.mark.parametrize("value_type, n_dim", motion_append_params)
def test_motion_easing(value_type, n_dim):
    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=[Easing.LINEAR])

    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))

    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=['linear'])

    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))

    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=[Easing.EASE_IN_OUT])

    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))

    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=['ease_in_out'])

    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 0.5) == np.array([0.5] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))

    motion = Motion(init_value=[0.0] * n_dim, value_type=value_type)
    motion.extend(keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=[lambda t: t])

    assert np.all(motion(None, 0.0) == np.array([0.0] * n_dim))
    assert np.all(motion(None, 1.0) == np.array([1.0] * n_dim))
