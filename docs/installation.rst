.. meta::
   :description: Install muse in an isolated Python environment using Miniforge and conda. Follow the setup guide for the MUSE solar synthesis package.

.. _muse-installing:

************
Installation
************

This page sets up Python and ``muse``.
For other methods, see :ref:`muse-installing-pypi` and sunpy's guide (:ref:`sunpy-topic-guide-installing`).

Installing Python
=================

Follow these steps even if you have Python: they create a virtual environment, an isolated place to install packages (like ``muse``) without affecting any other Python installation.
If you already use miniforge, skip the next section.

If you use Anaconda, we recommend replacing it with miniforge: `the default Anaconda channels have a restrictive license <https://sunpy.org/posts/2024/2024-08-09-anaconda/>`__, while miniforge sets up ``conda`` with the ``conda-forge`` channel, free for everyone.

.. _muse-installing-miniforge:

Installing miniforge (and conda)
================================

`Miniforge <https://github.com/conda-forge/miniforge/#miniforge>`__ installs ``conda`` with ``conda-forge``, the channel (a remote software repository) that has ``muse``, as its default.
Download the installer for your system and architecture:

.. grid:: 3

    .. grid-item-card:: Linux

        `x86-64 <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh>`__

        `aarch64 <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh>`__

        `ppc64le <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh>`__

    .. grid-item-card:: Windows
        :link: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

        `x86-64 <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>`__

    .. grid-item-card:: Mac

        `arm64 (Apple
        Silicon) <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>`__

        `x86-64 <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh>`__

Then select your platform to install miniforge:

.. tab-set::

    .. tab-item:: Linux & Mac
        :sync: platform

        Run the downloaded script:

        .. code-block:: console

            bash Miniforge3-$(uname)-$(uname -m).sh

        Then restart your terminal, or log out and in if the changes don't take effect.

    .. tab-item:: Windows
        :sync: platform

        Double click the downloaded executable.
        It adds a "Miniforge Prompt" entry to your start menu.

In a new terminal (Miniforge Prompt on Windows) run ``conda list`` to test that the install has worked.

Installing muse from conda-forge
================================

In a terminal (the Miniforge Prompt on Windows), create and activate a virtual environment, here named ``muse``:

.. code-block:: bash

    $ conda create --name muse
    $ conda activate muse

.. dropdown:: Click here if you haven't installed miniforge
    :color: warning

    If you cannot replace Anaconda or miniconda with miniforge (see `this blog post <https://sunpy.org/posts/2024/2024-08-09-anaconda/>`__ for why you should), make conda-forge the default channel:

    .. code-block:: bash

        $ conda config --add channels conda-forge
        $ conda config --set channel_priority strict

    These commands, from the `conda-forge documentation <https://conda-forge.org/docs/user/introduction/#how-can-i-install-packages-from-conda-forge>`__, affect every environment, the base one included.
    Do not install or upgrade packages in the base environment: a new environment for each project avoids the pitfalls of `multiple channels <https://conda-forge.org/docs/user/tipsandtricks/#multiple-channels>`__.

Then install ``muse``:

.. code-block:: bash

    $ conda install -c conda-forge musepy-lmsal

The conda-forge package is named ``musepy-lmsal``, but you import it in Python as ``muse``.
For Jupyter notebooks, also install ``ipywidgets`` and ``itables``.
Check the installation with:

.. code-block:: bash

    $ conda list musepy-lmsal

.. _muse-installing-pypi:

Installing muse from PyPI
=========================

If you manage your own Python (3.13 or later) and virtual environments, install ``muse`` from PyPI into an activated environment and check it:

.. code-block:: bash

    $ python -m pip install muse
    $ python -m pip show muse

.. _muse-installing-extras:

Optional extras
===============

``pip install "muse[<extra>]"`` adds:

* ``fiasco``: CHIANTI line lists and continua with `fiasco <https://fiasco.readthedocs.io/>`__, the default backend.
* ``chianti``: the same with ChiantiPy, which reads the database that ``XUVTOP`` points to.
* ``aia``: SDO/AIA effective areas.
* ``torch``: the Torch synthesis backend, see below.
* ``all``: all of these.

.. _muse-installing-torch:

Installing Torch (optional)
===========================

``muse`` synthesizes spectra with NumPy unless asked for the optional `PyTorch <https://pytorch.org/>`__ backend (``vdem_synthesis(..., backend="torch")``).
The right Torch build depends on your hardware (CPU-only, NVIDIA CUDA, or Apple Silicon): on Linux the generic PyPI wheel bundles the multi-gigabyte CUDA stack even without a GPU.

.. tab-set::

    .. tab-item:: conda

        conda-forge picks the right variant for your platform:

        .. code-block:: bash

            $ conda install pytorch

    .. tab-item:: pip

        Install the Torch build for your hardware from the `PyTorch install page <https://pytorch.org/get-started/locally/>`__ first; the extra then leaves it alone:

        .. code-block:: bash

            $ pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only example
            $ pip install "muse[torch]"

        On macOS and Windows ``pip install "muse[torch]"`` alone gives a CPU (and Apple Silicon ``mps``) build; the two steps are for Linux or a specific CUDA version.

    .. tab-item:: uv

        uv can detect your hardware and pick the matching build:

        .. code-block:: bash

            $ uv pip install torch --torch-backend=auto
            $ uv pip install "muse[torch]"

With ``muse`` installed, the :doc:`example gallery <generated/gallery/index>` walks through the synthesis pipeline end to end.
