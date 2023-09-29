.. module: movis.contrib

movis.contrib
==================

.. note::

   Please be advised that this module is experimental in nature.
   It is not imported automatically and does not enforce testing requirements.
   However, it is subject to active development and updates.

Overview
--------

The `movis.contrib` module serves as a sandbox for experimental functionalities and
features that are not yet part of the core Movis library.
It offers a wide range of utilities and tools that are under active development.
This module provides a platform for users to test new features before they are integrated into the main package.

Presentation
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.contrib.presentation.Slide
   movis.contrib.presentation.Character


Segmentation
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.contrib.segmentation.ChromaKey
   movis.contrib.segmentation.RobustVideoMatting


Voicevox
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.contrib.voicevox.make_voicevox_dataframe
   movis.contrib.voicevox.make_timeline_from_voicevox
   movis.contrib.voicevox.merge_timeline