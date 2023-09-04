from functools import partial

import movis as mv
import numpy as np


def palette(t: np.ndarray) -> np.ndarray:
    a = np.array([0.5, 0.5, 0.5])[:, None, None]
    b = np.array([0.5, 0.5, 0.5])[:, None, None]
    c = np.array([1.0, 1.0, 1.0])[:, None, None]
    d = np.array([0.263, 0.416, 0.557])[:, None, None]
    return a + b * np.cos(6.28318 * (c * t + d))


def length(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=0))


def fract(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def render(time: float, size: tuple[int, int], eps: float = 1e-8) -> np.ndarray:
    eps = 1e-8
    fragCoord = np.mgrid[:size[1], :size[0]]
    iResolution_xy = (np.array([size[1], size[0]]) - 1).reshape(2, 1, 1)
    iResolution_y = size[1]
    uv = (2 * fragCoord - iResolution_xy) / iResolution_y
    uv0 = uv.copy()
    finalColor = np.zeros((3, size[1], size[0]))

    for i in range(4):
        uv = fract(uv * 1.5) - 0.5
        d = length(uv) * np.exp(-length(uv0))
        col = palette(length(uv0) + i * 0.4 + time * 0.4)

        d = np.sin(d * 8 + time) / 8
        d = np.abs(d) + eps
        d = np.power(0.01 / d, 1.2)
        finalColor += col * d

    img = np.concatenate([
        np.clip(finalColor, 0, 1),
        np.full((1, *finalColor.shape[1:]), 1.0)])
    return (255 * img.transpose(1, 2, 0)).astype(np.uint8)


if __name__ == '__main__':
    # NOTE: This example is ported from https://www.youtube.com/watch?v=f4s1h2YETNY
    scene = mv.layer.Composition((1024, 576), 10.0)
    scene.add_layer(partial(render, size=scene.size))
    scene.write_video('shader_art_intro.mp4')
