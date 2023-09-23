Installation
=============

Movis is a pure python package, so it can be installed with pip:

.. code-block:: bash

    pip install movis

If you want to install the latest development version, you can install it from the git repository:

.. code-block:: bash

    pip install git+https://github.com/rezoo/movis.git


Movis relies on several relatively large packages.
Fortunately, prebuilt libraries for all of these are distributed on PyPI.
In most cases, you can install these dependencies effortlessly by utilizing the ``requirements.txt`` file.
If the installation fails, it's crucial to identify which package caused the failure and then
proceed with the following troubleshooting steps.


librosa
-------

`librosa <https://librosa.org/>`_ is a library for audio analysis.
You can usually install it via pip. Alternatively, you can use conda with the following command:

.. code-block:: bash

    conda install -c conda-forge librosa

If the installation doesn't work, refer to `librosa installation documentation <https://librosa.org/doc/main/install.html>`_.

OpenCV
-------

`OpenCV <https://opencv.org/>`_ is used for layer geometric transformations
and filter processing like blurring.
While building from source can maximize performance,
installing the prebuilt ``opencv-python`` package via pip is usually more convenient.
It is also available from conda-forge:

.. code-block:: bash

    conda install -c conda-forge opencv

.. note::

    Although installing OpenCV from ``conda-forge`` or ``pip`` usually installs the CPU-only version,
    movis currently does not utilize GPU acceleration.

If you encounter an ``ImportError`` related to OpenCV while trying to run Movis on Docker,
please add the following package installation to your ``Dockerfile``:

.. code-block:: bash

    sudo apt-get install -y libgl1-mesa-dev libglib2.0-0

PySide6
-------

`PySide6 <https://wiki.qt.io/Qt_for_Python>`_ is used for high-quality text and rectangle rendering.
In the author's experience, installing via pip usually works without issues, however,
if it fails, consider building from source, as described on `PySide6's PyPI page <https://pypi.org/project/PySide6/>`_.

.. note::

    Be aware that ``PySide6`` is an LGPL library.
    If you consider distributing Movis as a single binary,
    you'll need to comply with the LGPL, which requires source code disclosure.

imageio / imageio-ffmpeg
------------------------

`imageio <https://github.com/imageio/imageio>`_ and
`imageio-ffmpeg <https://github.com/imageio/imageio-ffmpeg>`_ are used for reading and writing videos.
The author has not experienced any installation failures in their environment, however,
the installation might fail in special environments.
For more details, visit `imageio-ffmpeg's GitHub page <https://github.com/imageio/imageio-ffmpeg>`_.

