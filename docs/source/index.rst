.. image:: ../../images/movis_logo.png

Movis: Video Editing as a Code
=================================

Movis is an engine written in Python, purposed for video production tasks.
This library allows users to generate various types of videos,
including but not limited to presentation videos, explainer videos,
training videos, and game commentary videos, through Python programming.

Contrary to many existing video production software solutions,
Movis does not offer a Graphical User Interface (GUI).
This may be perceived as a limitation for users new to video editing,
but it serves as an advantage for automation.
Specifically, engineers can integrate their own Artificial Intelligence
models to execute tasks such as facial image anonymization or
automatic summarization of videos based on detection of changes.
Additionally, the system is designed to work with highly
programmable interactive interfaces like LLM to facilitate
automated video editing processes.

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
