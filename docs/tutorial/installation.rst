.. _muse-tutorial-installing:

************
Installation
************

This is the first chapter in the ``muse`` tutorial, and by the end of it you should have a working installation of Python and ``muse``.
For further information and alternative methods for installing ``muse`` beyond the recommended approach outlined below, refer to sunpy's documentation (:ref:`sunpy-topic-guide-installing`).

Installing Python
=================

There are many ways to install Python, but even if you have Python installed somewhere on your computer we recommend following these instructions anyway.
That's because we will create a new Python virtual environment.
As well as containing a Python installation, this virtual environment provides an isolated place to install Python packages (like ``muse``) without affecting any other current Python installation.
If you already have Python and ``conda`` working you can skip the next section.

`If you are using Anaconda, we recommend that you uninstall it as the default package channel(s) have a restrictive license which means you might not be able to use it for free <https://sunpy.org/posts/2024/2024-08-09-anaconda/>`__.
Instead, we recommend that you use miniforge which is a minimal installer that set ups ``conda`` with the ``conda-forge`` channel, which is free to use for everyone.
If you are using miniforge, you can skip the next section.

.. _muse-tutorial-installing-miniforge:

Installing miniforge (and conda)
================================

If you don't already have a Python installation then we recommend installing Python with `miniforge <https://github.com/conda-forge/miniforge/#miniforge>`__.
Miniforge will install ``conda`` and automatically configure the default channel (a channel is a remote software repository) to be ``conda-forge``, which is where ``muse`` is available.

First, download the installer for your system and architecture from the links below:

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

        For Linux & Mac, run the downloaded script above using the following command:
        ``bash <filename>``. The following should work:

        .. code-block:: console

            bash Miniforge3-$(uname)-$(uname -m).sh

        Once the installer has completed, restart your terminal or log-out and log-in if the changes don't take effect.

    .. tab-item:: Windows
        :sync: platform

        Double click the executable file downloaded from the links above.

        Once the installer has completed you should have a new "Miniforge Prompt" entry in your start menu.

In a new terminal (Miniforge Prompt on Windows) run ``conda list`` to test that the install has worked.

Installing muse
===============

To install ``muse``, start by launching a terminal (under a UNIX-like system) or the Miniforge Prompt (under Windows).
Now we will create and activate a new virtual environment to install ``muse`` into:

.. code-block:: bash

    $ conda create --name muse
    $ conda activate muse

In this case the virtual environment is named 'muse'.
Feel free to change this to a different environment name.

The benefit of using a virtual environment is that it allows you to install packages without affecting any other Python installations or versions on your system.
This also means you can work on multiple projects (research or coding) with different package requirements without them interfering with each other.

.. dropdown:: Click here if you haven't installed miniforge
    :color: warning

    If you have installed miniforge or are using Anaconda you need to configure conda to get your packages from conda-forge as well as the defaults channel.

    You should no longer use the defaults channel at all, see `this blog post <https://sunpy.org/posts/2024/2024-08-09-anaconda/>`__ for details as to why.
    Therefore, if you are using Anaconda or miniconda we would suggest you uninstall it and install miniforge in its place.

    We also appreciate this isn't going to be possible for everyone, so what follows is our best instructions for how to proceed if you are using miniconda or Anaconda.

    The commands you need to run to add conda-forge and make it the default location to install conda packages from are:

    .. code-block:: bash

        $ conda config --add channels conda-forge
        $ conda config --set channel_priority strict

    These commands are taken from the
    `conda-forge documentation <https://conda-forge.org/docs/user/introduction/#how-can-i-install-packages-from-conda-forge>`__.

    Running these commands affect all the environments in your conda installation, critically, including the base Anaconda environment.
    We highly recommend that you do not install new packages, upgrade packages or use your base environment.
    Instead create new environments for all your projects, as you are much less likely to run into any pitfalls while using `multiple channels <https://conda-forge.org/docs/user/tipsandtricks/#multiple-channels>`__ by doing this.

Now that we have a fresh virtual environment, we can proceed with installing ``muse``:

.. code-block:: bash

    $ conda install muse

This will install ``muse`` and all of its dependencies.
If you are planning on using muse in jupyter notebooks we also recommend you install the ``ipywidgets`` and ``itables`` packages.

To ensure that ``muse`` was installed correctly, run the following command:

.. code-block:: bash

    $ conda list muse

This checks if ``muse`` was installed correctly.

If you want to install another package later, you can run ``conda install <package_name>``.

.. _muse-tutorial-installing-torch:

Installing Torch (optional)
===========================

``muse`` synthesizes spectra with NumPy by default; `PyTorch <https://pytorch.org/>`__ is an optional accelerator backend (``vdem_synthesis(..., backend="torch")``).
Torch is never selected implicitly and results do not change with what is installed, so you only need this section if you want the speed-up.

The right Torch build depends on your hardware (CPU-only, NVIDIA CUDA, or Apple Silicon), and the generic PyPI wheel is not always the one you want: on Linux it bundles the multi-gigabyte CUDA stack even on machines without a GPU.

.. tab-set::

    .. tab-item:: conda

        conda-forge picks the right variant for your platform:

        .. code-block:: bash

            $ conda install pytorch

    .. tab-item:: pip

        Install Torch first, following the selector on the `PyTorch install page <https://pytorch.org/get-started/locally/>`__ for your hardware, then install muse's extra; pip sees Torch is already satisfied and leaves it alone:

        .. code-block:: bash

            $ pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only example
            $ pip install "muse[torch]"

        On macOS and Windows the default ``pip install "muse[torch]"`` already gives a CPU (and Apple-Silicon ``mps``) build, so the two-step is only needed on Linux or for a specific CUDA version.

    .. tab-item:: uv

        uv can detect your hardware and pick the matching build:

        .. code-block:: bash

            $ uv pip install torch --torch-backend=auto
            $ uv pip install "muse[torch]"

Now we've got a working installation of ``muse``, in the next few chapters we'll look at some of the basic data structures ``muse`` uses for representing SG and context imager data.
