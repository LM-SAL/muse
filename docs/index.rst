.. meta::
   :description: Synthesize MUSE solar spectra from simulations with Python. Install muse, follow the synthesis tutorials, and explore the API reference.

.. _muse-index:

********************************************************
``muse``: Python tools for the Multi-slit Solar Explorer
********************************************************

``muse`` is an open-source Python package for the `Multi-slit Solar Explorer (MUSE) <https://muse.lmsal.com/>`__ mission.
Use it to build instrument responses, synthesize spectra from solar simulations, and analyze synthetic observations.
The package is pre-alpha; tools for reading and reducing mission observations are planned.

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

The package is pre-alpha and currently focuses on synthesizing MUSE observations from simulations:

* Build a Velocity-Differential Emission Measure (VDEM) from a simulation cube with :func:`muse.synthesis.create_simple_vdem`.
* Match a VDEM to the MUSE field of view and raster geometry with :func:`muse.transforms.match_fov`, :func:`muse.transforms.reshape_x_to_slit_step`, and :func:`muse.transforms.reshape_slit_step_to_x`.
* Create CHIANTI line lists (:func:`muse.instrument.create_chianti_line_list`) and build, save, and load spectrograph response functions (:func:`muse.instrument.create_spectral_response`, :func:`muse.instrument.save_response`, :func:`muse.instrument.read_response`, :func:`muse.instrument.load_and_concat_responses`).
* Synthesize detector spectra with :func:`muse.synthesis.vdem_synthesis` and analyze them with :func:`muse.synthesis.calculate_moments`.

The :doc:`example gallery <generated/gallery/index>` walks through this pipeline end to end.

Planned capabilities
====================

Support for reading, calibrating, and visualizing MUSE imaging and spectroscopic observations is planned but not yet implemented.
See the :ref:`observation-roadmap` for details and :doc:`muse` for an overview of the mission.

Citing muse
===========

If you use ``muse`` in a paper, please cite |DOI|.

Getting help
============

If you would like to get in touch with someone who works on ``muse`` **for any reason**, we suggest opening an issue on the `muse GitHub issue tracker <https://github.com/LM-SAL/muse/issues>`__.

The source code is available in the `muse GitHub repository <https://github.com/LM-SAL/muse>`__.

.. |DOI| image:: https://zenodo.org/badge/1168788621.svg
   :target: https://doi.org/10.5281/zenodo.22647387
