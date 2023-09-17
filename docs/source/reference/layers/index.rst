.. module: movis.layer

movis.layer
============

The :mod:`~movis.layer` module defines base protocols for representing various kind of video layers in :class:`~movis.layer.protocol.Layer` and :class:`~movis.layer.protocol.BasicLayer`.
The remaining classes in this module represent its implementations.

Composition
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.composition.Composition
   movis.layer.composition.LayerItem


Image, Video, Audio, etc.
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.media.Image
   movis.layer.media.ImageSequence
   movis.layer.media.Video
   movis.layer.media.Audio
   movis.layer.media.AudioSequence


Drawing layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.drawing.Line
   movis.layer.drawing.Rectangle
   movis.layer.drawing.Ellipse
   movis.layer.drawing.Text
   movis.layer.drawing.FillProperty
   movis.layer.drawing.StrokeProperty

Texture layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.texture.Gradient
   movis.layer.texture.Stripe

Layer-to-Layer Composition
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.layer_ops.AlphaMatte
   movis.layer.layer_ops.LuminanceMatte

Protocol
---------------

.. autoclass:: movis.layer.protocol.Layer
    :special-members: __call__

.. autoclass:: movis.layer.protocol.BasicLayer
    :members: __call__, duration, get_key

.. autoclass:: movis.layer.protocol.AudioLayer
    :members: __call__, get_audio