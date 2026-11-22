************
Contributing
************

``muse`` is open-source and community-developed, and welcomes feedback, bug reports, code and pull request reviews.

Providing Feedback
==================

Open an issue on the `issue tracker`_, comment in the `muse discussion`_, or email ``nfreij@seti.org``.
Criticism, questions about current or future capabilities, and praise are all welcome.

Reporting Bugs
==============

Report bugs and unexpected behavior on the `issue tracker`_, ideally with a short code example that reproduces them.
If you are not sure it is a bug, email ``nfreij@seti.org``.

Contributing Code
=================

Discuss your aims with the ``muse`` community before you start: it makes the work easier and avoids surprises, such as the feature already existing elsewhere.

Contributions go through `git`_ and `GitHub`_.
If they are new to you, see the `SunPy Newcomers Guide`_, the `GitHub guide`_ and the `git guide`_; the SunPy workflow applies to ``muse`` with the muse repository in place of sunpy's.
Seasoned developers can find more in the `sunpy Developers Guide`_.

Development environment
=======================

The test suite exercises every optional backend, so install the ``tests`` (or ``dev``) extra.
In a conda or micromamba environment named ``muse``:

.. code-block:: bash

    $ micromamba activate muse
    $ pip install -e ".[dev]"
    $ python -m pytest muse
    $ pre-commit run --all-files

Re-run the ``pip install`` when the extras in ``pyproject.toml`` change: the tests of a missing backend fail rather than skip.
On Linux without an NVIDIA GPU, first install the CPU Torch build (:ref:`muse-installing-torch`) so the ``dev`` extra does not pull the CUDA stack; tox and CI do this with ``UV_TORCH_BACKEND=cpu``.
``tox -e py314`` builds the locked environment of ``uv.lock`` and is the way to reproduce CI.

Continuous integration
======================

* **GitHub Actions** (``.github/workflows/ci.yaml``) runs the ``py313``/``py314`` tox tests, packaging (including an import of every package from an install without extras), the documentation build (``tox -e build_docs``), and the ``chianti``-marked tests against a downloaded CHIANTI database and the fiasco database built from it, both cached; that job fails if either backend or database is missing, and the workflow also runs on the 1st of each month.
* **CircleCI** (``.circleci/config.yml``) runs the figure-comparison tests and publishes their baseline images.

.. _issue tracker: https://github.com/LM-SAL/muse/issues
.. _SunPy Newcomers Guide: http://docs.sunpy.org/en/latest/dev_guide/newcomers.html
.. _GitHub: https://github.com/
.. _git: https://git-scm.com/
.. _GitHub guide: https://github.com/git-guides
.. _git guide: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
.. _sunpy Developers Guide: http://docs.sunpy.org/en/latest/dev_guide
.. _muse discussion: https://github.com/LM-SAL/muse/discussions
