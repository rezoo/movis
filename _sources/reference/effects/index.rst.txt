.. module: movis.effect


movis.effect
============

The :mod:`~movis.effect` module defines base protocols for representing video effects in
:class:`~movis.effect.protocol.Effect` and :class:`~movis.effect.protocol.BasicEffect`.
The remaining classes in this module represent its implementations.

Blur effects
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.effect.blur.GaussianBlur
   movis.effect.blur.Glow

Color correction effects
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.effect.color.FillColor
   movis.effect.color.HSLShift

Layer style effects
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.effect.style.DropShadow


Protocol
---------------

.. autoclass:: movis.effect.protocol.Effect
    :special-members: __call__

.. autoclass:: movis.effect.protocol.BasicEffect
    :members: get_key