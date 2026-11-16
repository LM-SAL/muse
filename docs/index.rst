.. meta::
   :description: Synthesize MUSE solar spectra from simulations with Python. Install muse, follow the synthesis tutorials, and explore the API reference.

.. _muse-index:

********************************************************
``muse``: Python tools for the Multi-slit Solar Explorer
********************************************************

``muse`` is an open-source Python package for the `Multi-slit Solar Explorer (MUSE) <https://muse.lmsal.com/>`__ mission.
Use it to build instrument responses, synthesize spectra from solar simulations, and analyze synthetic observations.

.. grid:: 1 1 3 3
    :gutter: 3

    .. grid-item-card:: Install
        :link: installation
        :link-type: doc

        Set up Python and install ``muse``.

    .. grid-item-card:: Examples
        :link: generated/gallery/index
        :link-type: doc

        Follow the synthesis workflow step by step.

    .. grid-item-card:: API reference
        :link: reference/index
        :link-type: doc

        Explore the package's functions and data model.

.. toctree::
    :hidden:
    :maxdepth: 1

    muse
    installation
    generated/gallery/index
    reference/index
    contributing
    design
    changelog

What ``muse`` can do today
==========================

The package is pre-alpha and synthesizes MUSE observations from simulations:

* Build a Velocity-Differential Emission Measure (VDEM) from a simulation cube with :func:`muse.synthesis.create_simple_vdem`.
* Match a VDEM to the MUSE field of view and raster geometry with :func:`muse.transforms.match_fov`, :func:`muse.transforms.reshape_x_to_slit_step`, and :func:`muse.transforms.reshape_slit_step_to_x`.
* Create CHIANTI line lists and continua with fiasco or ChiantiPy, and build, save and load spectrograph and context-imager responses (:func:`muse.instrument.create_spectral_response`, :func:`muse.instrument.read_response`), or all of them at once with :func:`muse.instrument.generate_sg_responses`.
* Synthesize detector spectra with :func:`muse.synthesis.vdem_synthesis`, convert their units with :func:`muse.instrument.transform_flux_units`, and analyze them with :func:`muse.synthesis.calculate_moments`.

The :doc:`example gallery <generated/gallery/index>` walks through this pipeline end to end.

Planned capabilities
====================

Reading, calibrating, and visualizing MUSE observations is planned; see the :ref:`observation-roadmap`.
:doc:`muse` gives an overview of the mission.

Citing muse
===========

If you use ``muse`` in a paper, please cite |DOI|.

Getting help
============

To reach the developers **for any reason**, open an issue on the `muse GitHub issue tracker <https://github.com/LM-SAL/muse/issues>`__.
The source code is in the `muse GitHub repository <https://github.com/LM-SAL/muse>`__.

.. |DOI| image:: https://zenodo.org/badge/1168788621.svg
   :target: https://doi.org/10.5281/zenodo.22647387
