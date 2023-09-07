.. module: movis.layer

movis.layer
============

The :mod:`~movis.layer` module defines base protocols for representing various kind of video layers in :class:`~movis.layer.protocol.Layer` and :class:`~movis.layer.protocol.BasicLayer`.
The remaining classes in this module represent its implementations.

Implementations
----------------

Composition
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.composition.Composition
   movis.layer.composition.LayerItem


Image file, Video file, etc.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.media.Image
   movis.layer.media.ImageSequence
   movis.layer.media.Video


Drawing layers
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.layer.drawing.Line
   movis.layer.drawing.Rectangle
   movis.layer.drawing.Ellipse
   movis.layer.drawing.Text


Layer protocol
---------------

.. autoclass:: movis.layer.protocol.Layer
    :special-members: __call__

.. autoclass:: movis.layer.protocol.BasicLayer
    :members: duration, get_key