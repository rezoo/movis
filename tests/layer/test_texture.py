import numpy as np
import pytest

import movis as mv


def test_gradient_init():
    grad = mv.layer.Gradient(size=(100, 100), start_color=(0, 0, 0), end_color=(255, 255, 255), duration=10)
    assert grad.size == (100, 100)
    assert np.all(grad.start_color(0.0) == (0, 0, 0))
    assert np.all(grad.end_color(0.0) == (255, 255, 255))
    assert grad.duration == 10


def test_gradient_invalid_type():
    with pytest.raises(ValueError):
        mv.layer.Gradient(gradient_type='invalid_type')


def test_gradient_call():
    grad = mv.layer.Gradient(
        size=(100, 100), start_color=(0, 0, 0), end_color=(255, 255, 255), duration=10)

    output = grad(5)
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape == (100, 100, 4)
    assert output.dtype == np.uint8

    start_color_pixel = output[0, 0]
    end_color_pixel = output[-1, -1]

    assert np.array_equal(start_color_pixel, [0, 0, 0, 255])
    assert np.array_equal(end_color_pixel, [255, 255, 255, 255])


def test_gradient_out_of_duration():
    grad = mv.layer.Gradient(size=(100, 100), start_color=(0, 0, 0), end_color=(255, 255, 255), duration=10)
    output = grad(15)

    assert output is None


def test_stripe_init():
    stripe = mv.layer.Stripe(size=(100, 100), color1=(0, 0, 0), color2=(255, 255, 255), angle=45.0)
    assert stripe.size == (100, 100)
    assert np.all(stripe.color1(0.0) == (0, 0, 0))
    assert np.all(stripe.color2(0.0) == (255, 255, 255))
    assert stripe.angle(0.0) == 45.0


def test_stripe_call():
    stripe = mv.layer.Stripe(size=(100, 100), color1=(0, 0, 0), color2=(255, 255, 255), angle=45.0)
    output = stripe(5)
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape == (100, 100, 4)
    assert output.dtype == np.uint8


def test_stripe_out_of_duration():
    stripe = mv.layer.Stripe(size=(100, 100), color1=(0, 0, 0), color2=(255, 255, 255), angle=45.0, duration=10)
    output = stripe(15)
    assert output is None
