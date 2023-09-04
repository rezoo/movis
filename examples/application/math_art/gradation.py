from functools import partial

import movis as mv
import numpy as np


def render(time: float, size: tuple[int, int]) -> np.ndarray:
    uv = np.mgrid[:size[1], :size[0]] / (np.array([size[1], size[0]]) - 1).reshape(2, 1, 1)
    col = 0.5 + 0.5 * np.cos(time + np.stack([uv[0], uv[1], uv[0]]) + np.array([0, 2, 4])[:, None, None])
    img = np.concatenate([col, np.full((1, *col.shape[1:]), 1.0)])
    return np.round(img * 255).astype(np.uint8).transpose(1, 2, 0)


if __name__ == '__main__':
    # NOTE: This example is ported from https://www.youtube.com/watch?v=f4s1h2YETNY&t=552s
    scene = mv.layer.Composition((1024, 576), 10.0)
    scene.add_layer(partial(render, size=scene.size))
    scene.write_video('gradation.mp4')
