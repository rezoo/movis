.. image:: ../../images/movis_logo.png

Movis: Video Editing as a Code
=================================

What is Movis?
---------------

Movis is an engine written in Python, purposed for video production tasks.
This library allows users to generate various types of videos,
including but not limited to presentation videos, explainer videos,
training videos, and game commentary videos, through Python.

Library without GUI for automation
----------------------------------

Contrary to many existing video production software solutions,
Movis does not offer a Graphical User Interface (GUI).
This may be perceived as a limitation for users new to video editing,
but it serves as an advantage for automation.
Specifically, engineers can
*integrate their own ML models to execute tasks* such as
facial image anonymization or automatic summarization of videos.
Additionally, this library works with highly programmable interactive
interfaces like LLMs to facilitate automated video editing processes.

Creating Video with Compositions
--------------------------------

Similar to other video editing software,
Movis employs the concept of "compositions" as the fundamental unit for video editing.
Within a composition, users can include multiple layers and manipulate
these layers' attributes over a time scale to produce a video.
Effects can also be selectively applied to these layers as needed.

Here's some example code:

.. code-block:: python

    import movis as mv

    scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
    scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
    scene.add_layer(
        mv.layer.Text('Hello World!', font_size=256, font_family='Helvetica', color='#ffffff'),
        name='text')
    scene['text'].add_effect(mv.effect.DropShadow(offset=10.0))
    scene['text'].scale.enable_motion().extend(
        keyframes=[0.0, 1.0], values=[0.0, 1.0], motion_types=['ease_in_out'])
    scene['text'].opacity.enable_motion().extend([0.0, 1.0], [0.0, 1.0])

    scene.write_video('output.mp4')

The composition can also be used as a layer.
By combining multiple compositions and layers, users can create complex videos.

Simple implementation of custom layers and effects
---------------------------------------------------

Movis is engineered to facilitate the straightforward implementation of user-defined layers,
thereby enabling the seamless integration of unique visual effects into video projects.
This design obviates the necessity for users to possess
intricate knowledge of the library or to become proficient
in advanced programming languages exemplified by C++.
Thus, users may focus their attention predominantly on
their creative concepts.
In instances where accelerated computational performance is requisite,
one may employ separate, specialized libraries such as
Jax or PyTorch to execute computations at an elevated speed via a GPU.

For example, to implement a user-defined layer, you only need to create a function that, given a time,
returns an `np.ndarray` with a shape of `(H, W, 4)` and dtype of `np.uint8` in RGBA order, or returns `None`.

.. code-block:: python

    import numpy as np
    import movis as mv

    size = (640, 480)

    def get_radial_gradient_image(time: float) -> np.ndarray:
        if time < 0.:
            return None
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

If you want to specify the duration of a layer,
the `duration` property is required. Movis also offers caching features
to accelerate rendering. If you wish to speed up rendering for layers
where the frame remains static, you can implement the `get_key(time: float)` method:

.. code-block:: python

    class RadialGradientLayer:
        def __init__(self, size: tuple[int, int], duration: float):
            self.size = size
            self.duration = duration
            self.center = np.array([size[1] // 2, size[0] // 2])
        
        def get_key(self, time: float) -> Hashable:
            # Returns 0 since the same image is always returned
            return 0
        
        def __call__(self, time: float) -> None | np.ndarray:
            # ditto.

Custom effects
^^^^^^^^^^^^^^

Effects for layers can also be implemented in a similar straightforward manner.

.. code-block:: python

    import cv2
    import movis as mv
    import numpy as np

    def apply_gaussian_blur(prev_image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(prev_image, ksize=(7, 7))

    scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
    scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
    scene.add_layer(
        mv.layer.Text('Hello World!', font_size=256, font_family='Helvetica', color='#ffffff'),
        name='text')
    scene['text'].add_effect(apply_gaussian_blur)


Reference
===========

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
