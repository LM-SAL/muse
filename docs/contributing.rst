**************************
Contributing to ``muse``
**************************

``muse`` is open-source and community-developed, and we are always glad to welcome new contributors and users.
You can contribute in several ways: by providing feedback, reporting bugs, contributing code, and reviewing pull requests.
There is a role for almost any level of engagement.

Providing Feedback
==================

We could always use more voices and opinions in the discussions about ``muse`` and its development from both users and developers.
There are several ways to make your voice heard.
You can open issues on our `issue tracker`_, comment in the `muse discussion`_, or email us directly at ``nfreij@seti.org``.
Whether it be (non)constructive criticism, inquiries about current or future capabilities, or flattering praise, we would love to hear from you.

Reporting Bugs
==============

If you run into unexpected behavior or a bug please report it.
All bugs are raised and stored on our `issue tracker`_.
If you are not sure whether your problem is a bug, a deficiency in functionality, or something else, you can email ``nfreij@seti.org``.
Ideally, we would like a short code example so we can run into the bug on our own machines.

Contributing Code
=================

If you would like to contribute code, it is strongly recommended that you first discuss your aims with the ``muse`` community.
We strive to be an open and welcoming community for developers of all experience levels.
Discussing your ideas before you start can give you new insights that will make your development easier, lead to a better end product, and reduce the chances of your work being regretfully rejected because of an issue you weren't aware of (e.g., the functionality already exists elsewhere).

In the rest of this section we will go through the steps needed to set up your system so you can contribute code to ``muse``.
This is done using `git`_ version control software and `GitHub`_, a website that allows you to upload, update, and share code repositories (repos).
If you are new to code development or git and GitHub you can learn more from the following guides:

* `SunPy Newcomers Guide`_
* `GitHub guide`_
* `git guide`_

The principles in the SunPy guides for contributing code and utilizing GitHub and git are exactly the same for ``muse``, except that we contribute to the muse repository rather than the ``sunpy`` one.
If you are a more seasoned developer and would like to get further information, you can check out the `sunpy Developers Guide`_.

Before you can contribute code to muse, you first need to install the development version of ``muse``.

Development environment
=======================

The full test suite intentionally exercises every optional backend, so your development environment must match the ``tests`` (or ``dev``) extra.
With a conda/micromamba environment named ``muse``:

.. code-block:: bash

    $ micromamba activate muse
    $ pip install -e ".[dev]"
    $ python -m pytest muse
    $ pre-commit run --all-files

Re-run the ``pip install`` whenever the extras in ``pyproject.toml`` change; if an optional backend (e.g. Torch) is missing, the corresponding tests will fail rather than silently skip.
On Linux without an NVIDIA GPU, install the slim CPU Torch build first (see :ref:`muse-tutorial-installing-torch`) so the ``dev`` extra does not pull the multi-gigabyte CUDA stack; tox environments and CI already do this via ``UV_TORCH_BACKEND=cpu``.
Alternatively, ``tox -e py314`` builds a complete, locked environment (from ``uv.lock``) and is the canonical way to reproduce CI results.

Continuous integration
======================

Two CI systems share the work:

* **GitHub Actions** (``.github/workflows/ci.yaml``) owns platform tests (the ``py312``/``py313``/``py314`` tox environments), packaging, documentation builds (``tox -e build_docs``), and the CHIANTI integration job, which downloads and caches the CHIANTI database to run the ``remote_data``-gated tests.
* **CircleCI** (``.circleci/config.yml``) owns the deterministic figure-comparison tests and publishes the reference images used as baselines.

.. _issue tracker: https://github.com/LM-SAL/muse/issues
.. _SunPy Newcomers Guide: http://docs.sunpy.org/en/latest/dev_guide/newcomers.html
.. _GitHub: https://github.com/
.. _git: https://git-scm.com/
.. _GitHub guide: https://github.com/git-guides
.. _git guide: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
.. _sunpy Developers Guide: http://docs.sunpy.org/en/latest/dev_guide
.. _muse discussion: https://github.com/LM-SAL/muse/discussions
