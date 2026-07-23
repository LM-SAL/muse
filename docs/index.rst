.. _muse-index:

**********************
``muse`` documentation
**********************

``muse`` is an open-source Python package for the `Multi-slit Solar Explorer (MUSE) <https://muse.lmsal.com/>`__ mission.
MUSE has not launched yet, so there are no observations to read; for now the package focuses on synthesizing MUSE observations of the main spectral lines from simulations.
Once MUSE data become available, `they will be publicly accessible together with co-aligned SDO/AIA data <https://muse.lmsal.com/search/>`__, and this package will provide the tools to read, manipulate, and visualize them.

What ``muse`` can do today
==========================

The package is pre-alpha and currently focuses on synthesizing MUSE observations from simulations:

* Build a Velocity-Differential Emission Measure (VDEM) from a simulation cube with :func:`muse.synthesis.create_simple_vdem`.
* Match a VDEM to the MUSE field of view and raster geometry with :func:`muse.transforms.match_fov`, :func:`muse.transforms.reshape_x_to_slit_step`, and :func:`muse.transforms.reshape_slit_step_to_x`.
* Create CHIANTI line lists (:func:`muse.instrument.create_chianti_line_list`) and build, save, and load spectrograph response functions (:func:`muse.instrument.create_spectral_response`, :func:`muse.instrument.save_response`, :func:`muse.instrument.read_response`, :func:`muse.instrument.load_and_concat_responses`).
* Synthesize detector spectra with :func:`muse.synthesis.vdem_synthesis` and analyze them with :func:`muse.synthesis.calculate_moments`.

The :doc:`example gallery <generated/gallery/index>` walks through this pipeline end to end.

Roadmap (not yet implemented)
=============================

The longer-term goal of ``muse`` is to provide a set of classes for handling both imaging (context imager) and spectral observations (spectrograph).
The classes will link the observations with various forms of supporting data including: measurement uncertainties; units; a data mask to mark pixels with unreliable or unphysical data values; WCS (World Coordinate System) transformations that describe the position, wavelengths, and times represented by the pixels; and general metadata.
These classes will also provide methods for applying a number of calibration routines including exposure time correction and conversion between data number, photons, and energy units, referred to as radiometric calibration.
Furthermore, it will allow you to plug in your own custom calibration routines and apply them to the level 2 data to generate level 3 data.
None of this observation-handling functionality exists yet.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :class-card: card

        Getting started
        ^^^^^^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          muse
          tutorial/index
          generated/gallery/index

    .. grid-item-card::
        :class-card: card

        Other info
        ^^^^^^^^^^
        .. toctree::
          :maxdepth: 1

          reference/index
          known_issues
          contributing
          design
          changelog

Getting help
============

If you would like to get in touch with someone who works on ``muse`` **for any reason**, we suggest opening an issue on the `muse GitHub issue tracker <https://github.com/LM-SAL/muse/issues>`__.
