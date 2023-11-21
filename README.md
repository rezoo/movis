![GitHub Logo](images/movis_logo.png)

# Movis: Video Editing as a Code

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/movis.svg)](https://pypi.python.org/pypi/movis)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rezoo/movis)
![Continuous integration](https://github.com/rezoo/movis/actions/workflows/python-package.yml/badge.svg)
![Docs](https://github.com/rezoo/movis/actions/workflows/docs.yml/badge.svg)

[**Docs**](https://rezoo.github.io/movis/)
| [**Overview**](https://rezoo.github.io/movis/overview.html)
| [**Install Guide**](https://rezoo.github.io/movis/install.html)
| [**Examples**](https://github.com/rezoo/movis/tree/main/examples)
| [**API Reference**](https://rezoo.github.io/movis/reference/index.html)
| [**Contribution Guide**](https://rezoo.github.io/movis/contribution.html)

## ✅ What is Movis?

Movis is an engine written in Python, purposed for video production tasks.
This library allows users to generate various types of videos,
including but not limited to presentation videos, motion graphics,
shader art coding, and game commentary videos, through Python.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezoo/movis/blob/main/docs/Quickstart.ipynb)

## 🚀 Main Features

* Easy and intuitive video editing (including scene cut, transition, crop, concatenation, inserting images and texts, etc.)
* Layer transformation (position, scale, and rotation) with sub-pixel precision
* Support for a variety of Photoshop-level blending modes
* Keypoint and easing-based animation engine
* Nested compositions
* Inserting text layers containing multiple outlines
* Simple audio editing (including fade-in and fade-out effects)
* Support for a variety of video and audio formats using ffmpeg
* Layer effects (drop shadow, grow, blur, chromakey, etc.)
* Support for rendering at 1/2 quality and 1/4 quality for drafts
* Fast rendering using cache mechanism
* Adding user-defined layers, effects, and animations without using inheritance

## 💻 Installation

Movis is a pure Python library and can be installed via the [Python Package Index](https://pypi.org/):

```bash
$ pip install movis
```

We have confirmed that it works with Python 3.9 to 3.11.

## ⭐️ Code Overview

### Creating Video with Compositions

Similar to other video editing software,
Movis employs the concept of *"compositions"* as the fundamental unit for video editing.
Within a composition, users can include multiple layers and manipulate
these layers' attributes over a time scale to produce a video.
Effects can also be selectively applied to these layers as needed.

Here's some example code:

```python
import movis as mv

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))  # Set background

pos = scene.size[0] // 2, scene.size[1] // 2
scene.add_layer(
    mv.layer.Text('Hello World!', font_size=256, font_family='Helvetica', color='#ffffff'),
    name='text',  # The layer item can be accessed by name
    offset=1.0,  # Show the text after one second
    position=pos,  # The layer is centered by default, but it can also be specified explicitly
    anchor_point=(0.0, 0.0),
    opacity=1.0, scale=1.0, rotation=0.0,  # anchor point, opacity, scale, and rotation are also supported
    blending_mode='normal')  # Blending mode can be specified for each layer.
scene['text'].add_effect(mv.effect.DropShadow(offset=10.0))  # Multiple effects can be added.
scene['text'].scale.enable_motion().extend(
    keyframes=[0.0, 1.0], values=[0.0, 1.0], easings=['ease_in_out'])
# Fade-in effect. It means that the text appears fully two seconds later.
scene['text'].opacity.enable_motion().extend([0.0, 1.0], [0.0, 1.0])

scene.write_video('output.mp4')
```

The composition can also be used as a layer.
By combining multiple compositions and layers, users can create complex videos.

```python
scene2 = mv.layer.Composition(scene.size, duration=scene.duration)
layer_item = scene2.add_layer(scene, name='scene')
# Equivalent to scene2['scene'].add_effect(...)
layer_item.add_effect(mv.effect.GaussianBlur(radius=10.0))
```

### Simple video processing

Of course, movis also supports simple video processing such as video merging and trimming.

#### concat

```python
intro = mv.layer.Video('intro.mp4')
title = mv.layer.Video('title.mp4')
chapter1 = mv.layer.Composition(size=(1920, 1080), duration=60.0)
...
scene = mv.concatenate([intro, title, chapter1, ...])
scene.write_video('output.mp4')
```

#### cutout

```python
raw_video = mv.layer.Video('video.mp4')
# select 0.0-1.0 secs and 2.0-3.0 secs, and concatenate them
scene = mv.trim(layer, start_times=[0.0, 2.0], end_times=[1.0, 3.0])
scene.write_video('output.mp4')
```

#### cropping

```python
layer = mv.layer.Image("image.png", duration=1.0)
# crop from x, y = (10, 20) with size w, h = (100, 200)
layer = mv.crop(layer, (10, 20, 100, 200))
```

#### fade-in / out

```python
layer = mv.layer.Video('video.mp4')
video1 = mv.fade_in(layer, 1.0)  # fade-in for 1.0 secs
video2 = mv.fade_out(layer, 1.0)  # fade-out for 1.0 secs
video3 = mv.fade_in_out(layer, 1.0, 2.0)  # fade-in for 1.0 secs and fade-out for 2.0 secs
```

### Implementation of custom layers, effects, and animations

Movis is designed to make it easy for users to implement custom layers and effects.
This means that engineers can easily integrate their preferred visual effects and animations using Python.

For example, let's say you want to create a demo video using your own machine learning model for tasks
like anonymizing face images or segmenting videos.
With Movis, you can easily do this without the need for more complex languages like C++,
by directly using popular libraries such as [PyTorch](https://pytorch.org/) or [Jax](https://github.com/google/jax).
Additionally, for videos that make use of GPGPU like [shader art](https://www.shadertoy.com/),
you can implement these intuitively through Python libraries like [Jax](https://github.com/google/jax) or [cupy](https://cupy.dev/).

For example, to implement a user-defined layer, you only need to create a function that, given a time,
returns an `np.ndarray` with a shape of `(H, W, 4)` and dtype of `np.uint8` in RGBA order, or returns `None`.

```python
import numpy as np
import movis as mv

size = (640, 480)

def get_radial_gradient_image(time: float) -> np.ndarray:
    center = np.array([size[1] // 2, size[0] // 2])
    radius = min(size)
    inds = np.mgrid[:size[1], :size[0]] - center[:, None, None]
    r = np.sqrt((inds ** 2).sum(axis=0))
    p = 255 - (np.clip(r / radius, 0, 1) * 255).astype(np.uint8)
    img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    img[:, :, :3] = p[:, :, None]
    img[:, :, 3] = 255
    return img

scene = mv.layer.Composition(size, duration=5.0)
scene.add_layer(get_radial_gradient_image)
scene.write_video('output.mp4')
```

If you want to specify the duration of a layer,
the `duration` property is required. Movis also offers caching features
to accelerate rendering. If you wish to speed up rendering for layers
where the frame remains static, you can implement the `get_key(time: float)` method:

```python
class RadialGradientLayer:
    def __init__(self, size: tuple[int, int], duration: float):
        self.size = size
        self.duration = duration
        self.center = np.array([size[1] // 2, size[0] // 2])
    
    def get_key(self, time: float) -> Hashable:
        # Returns 1 since the same image is always returned
        return 1
    
    def __call__(self, time: float) -> None | np.ndarray:
        # ditto.
```

#### Custom effects

Effects for layers can also be implemented in a similar straightforward manner.

```python
import cv2
import movis as mv
import numpy as np

def apply_gaussian_blur(prev_image: np.ndarray, time: float) -> np.ndarray:
    return cv2.GaussianBlur(prev_image, (7, 7), -1)

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
scene.add_layer(
    mv.layer.Text('Hello World!', font_size=256, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].add_effect(apply_gaussian_blur)
```

#### User-defined animations

Animation can be set up on a keyframe basis, but in some cases,
users may want to animate using user-defined functions.
movis provides methods to handle such situations as well.

```python
import movis as mv
import numpy as np

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(
    mv.layer.Text('Hello World!', font_size=256, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].position.add_function(
    lambda prev_value, time: prev_value + np.array([0, np.sin(time * 2 * np.pi) * 100]),
)
```

### Fast Prototyping on Jupyter Notebook

Jupyter notebooks are commonly used for data analysis that requires a lot of trial and error using Python.
Some methods for Jupyter notebooks are also included in movis to speed up the video production process.

For example, ``composition.render_and_play()`` is often used to
preview a section of a video within a Jupyter notebook.

```python
import movis as mv

scene = mv.layer.Composition(size=(1920, 1080), duration=10.0)
... # add layers and effects...
scene.render_and_play(
    start_time=5.0, end_time=10.0, preview_level=2)  # play the video from 5 to 10 seconds
```

This method has an argument called ``preview_level``.
For example, by setting it to 2, you can sacrifice video quality
by reducing the final resolution to 1/2 in exchange for faster rendering.

If you want to reduce the resolution when exporting videos or still images using
``composition.write_video()`` or similar methods,
you can use the syntax ``with composition.preview(level=2)``.

```python
import movis as mv

scene = mv.layer.Composition(size=(1920, 1080), duration=10.0)
... # add layers and effects...
with scene.preview(level=2):
    scene.write_video('output.mp4')  # The resolution of the output video is 1/2.
    img = scene(5.0)  # retrieve an image at t = 5.0
assert img.shape == (540, 960, 4)
```

Within this scope, the resolution of all videos and images will be reduced to 1/2.
This can be useful during the trial and error process.

## 📃 License

MIT License (see `LICENSE` for details).
