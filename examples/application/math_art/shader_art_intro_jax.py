from functools import partial

import movis as mv
import numpy as np
import jax
import jax.numpy as jnp


def palette(t: jax.Array) -> jax.Array:
    a = jnp.array([0.5, 0.5, 0.5])[:, None, None]
    b = jnp.array([0.5, 0.5, 0.5])[:, None, None]
    c = jnp.array([1.0, 1.0, 1.0])[:, None, None]
    d = jnp.array([0.263, 0.416, 0.557])[:, None, None]
    return a + b * jnp.cos(6.28318 * (c * t + d))


def length(x: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(x * x, axis=0))


def fract(x: jax.Array) -> jax.Array:
    return x - jnp.floor(x)


@partial(jax.jit, static_argnames=('size', 'eps'))
def _render(time: float, size: tuple[int, int], eps: float = 1e-8) -> jax.Array:
    fragCoord = jnp.mgrid[:size[1], :size[0]]
    iResolution_xy = (jnp.array([size[1], size[0]]) - 1).reshape(2, 1, 1)
    iResolution_y = size[1]
    uv = (2 * fragCoord - iResolution_xy) / iResolution_y
    uv0 = uv.copy()
    finalColor = jnp.zeros((3, size[1], size[0]))

    for i in range(4):
        uv = fract(uv * 1.5) - 0.5
        d = length(uv) * jnp.exp(-length(uv0))
        col = palette(length(uv0) + i * 0.4 + time * 0.4)

        d = jnp.sin(d * 8 + time) / 8
        d = jnp.abs(d) + eps
        d = jnp.power(0.01 / d, 1.2)
        finalColor += col * d

    img = jnp.concatenate([
        jnp.clip(finalColor, 0, 1),
        jnp.full((1, *finalColor.shape[1:]), 1.0)])
    return (255 * img.transpose(1, 2, 0)).astype(np.uint8)


def render(time: float, size: tuple[int, int], eps: float = 1e-8) -> np.ndarray:
    return np.asarray(_render(time, size, eps))


if __name__ == '__main__':
    # NOTE: This example is ported from https://www.youtube.com/watch?v=f4s1h2YETNY
    scene = mv.layer.Composition((1024, 576), 10.0)
    scene.add_layer(partial(render, size=scene.size))
    scene.write_video('shader_art_intro.mp4')
