.. movis documentation master file, created by
   sphinx-quickstart on Sun Sep  3 14:32:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Movis: Video Editing as a Code
=================================

Movis is a Python engine designed for video production. With this library, users can create a wide range of videos through Python, including presentation videos, explainer videos, training videos, and game commentary videos.

Unlike many other video production softwares, Movis doesn't include a GUI. While this might be a drawback for beginners, it is advantageous for automating video production tasks. Specifically, engineers can use their own AI models to automate processes such as anonymizing facial images, or generating summary videos by detecting points of change within a video. Additionally, by leveraging interactive interfaces with high programming affinity like LLM, one can also automate the video editing process.

Much like other video production software, Movis employs "compositions" as the basic unit for editing. One can add multiple layers to a composition and animate each layer's attributes over a timeline to create a video. Effects can also be applied to the target layers as needed.

Here's some example code:

.. code-block:: python

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


Reference
===========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
