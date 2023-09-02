<div align="center">
<img src="https://github.com/rezoo/movis/blob/main/images/movis_logo.png?raw=true" width="800" alt="logo"></img>
</div>

# Movis: Video Editing as a Code

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rezoo/movis)
![Continuous integration](https://github.com/rezoo/movis/actions/workflows/python-package.yml/badge.svg)

## What is Movis?

Movis is a Python engine designed for video production. With this library, users can create a wide range of videos through Python, including presentation videos, explainer videos, training videos, and game commentary videos.

#### Library without GUI for automation

Unlike many other video production softwares, Movis doesn't include a GUI. While this might be a drawback for beginners, it is advantageous for automating video production tasks. Specifically, engineers can use their own AI models to automate processes such as anonymizing facial images, or generating summary videos by detecting points of change within a video. Additionally, by leveraging interactive interfaces with high programming affinity like LLM, one can also automate the video editing process.

#### Compositions

Much like other video production software, Movis employs "compositions" as the basic unit for editing. One can add multiple layers to a composition and animate each layer's attributes over a timeline to create a video. Effects can also be applied to the target layers as needed.

Here's some example code:

```python
import movis as mv

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
scene.add_layer(
    mv.text.Text('Hello World!', font_size=128, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].add_effect(mv.layer.DropShadow(offset=5.0))
scene['text'].scale.enable_motion().extend(
    keyframes=[0.0, 1.0], values=[0.0, 1.0], motion_types=['ease_in_out'])
scene['text'].opacity.enable_motion().extend([0.0, 1.0], [0.0, 1.0])

scene.write_video('output.mp4')
```

The composition can also be used as a layer. By combining multiple compositions and layers, users can create complex videos.

## Simple implementation of custom layers and effects

### Custom layers

Movis allows you to add custom layers and effects written in Python. The requirements for implementing a layer are simple: you only need to create a function that, given a time, returns an `np.ndarray` with a shape of `(H, W, 4)` and dtype of `np.uint8` in RGBA order, or returns `None`.

```python
import numpy as np
import movis as mv

size = (640, 480)

def get_radial_gradient_image(time: float) -> None | np.ndarray:
    if time < 0.:
        return None
    center = np.array([size[0] // 2, size[1] // 2])
    radius = min(size)
    inds = np.mgrid[:size[1], :size[0]] - center[:, None, None]
    r = np.sqrt((inds ** 2).sum(axis=0))
    p = (np.clip(r / radius, 0, 1) * 255).astype(np.uint8)
    img = np.zeros(size[1], size[0], 4, dype=np.uint8)
    img[:, :, :3] = p[:, :, None]
    img[:, :, 3] = 255
    return img

scene = mv.layer.Composition(size, duration=5.0)
scene.add_layer(get_radial_gradient_image)
scene.write_video('output.mp4')
```

If you want to specify the duration of a layer, the `duration` property is required. Movis also offers caching features to accelerate rendering. If you wish to speed up rendering for layers where the frame remains static, you can implement the `get_key(time: float)` method:

```python
class RadialGradientLayer:
    def __init__(self, size: tuple[int, int], duration: float):
        self.size = size
        self.duration = duration
        self.center = np.array([size[0] // 2, size[1] // 2])
    
    def get_key(self, time: float) -> Hashable:
        # Returns 0 since the same image is always returned
        return 0
    
    def __call__(self, time: float) -> None | np.ndarray:
        # ditto.
```

### Custom effects

Effects for layers can also be implemented in a similar straightforward manner.

```python
def apply_gaussian_blur(prev_image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(prev_image, ksize=(7, 7))

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
scene.add_layer(
    mv.text.Text('Hello World!', font_size=128, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].add_effect(apply_gaussian_blur)
```

## Installation

Movis is a pure Python library and can be installed via the Python Package Index:

```bash
# PyPI
$ pip install movis
```

We have confirmed that movis works with Python versions 3.9 to 3.11.

## License

MIT License (see `LICENSE` for details). However, please note that movis uses PySide6 for some modules, which is under the LGPL.
