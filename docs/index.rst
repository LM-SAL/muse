.. _muse-index:

**********************
``muse`` documentation
**********************

``muse`` is an open-source Python package that provides tools to read, manipulate, and visualize `ulti-slit Solar Explorer (MUSE) <https://muse.lmsal.com/>`__ data.
`The data is publicly available and provides access to co-aligned SDO/AIA data and more. <https://muse.lmsal.com/search/>`__

The goal of ``muse`` is to provide a set of classes for handling both imaging (slit-jaw) and spectral observations (spectrograph).
The classes link the observations with various forms of supporting data including: measurement uncertainties; units; a data mask to mark pixels with unreliable or unphysical data values; WCS (World Coordinate System) transformations that describe the position, wavelengths, and times represented by the pixels; and general metadata.
These classes also provide methods for applying a number of calibration routines including exposure time correction and conversion between data number, photons, and energy units, referred to as radiometric calibration.
Furthermore, it allows you to plug in your own custom calibration routines and apply them to the level 2 data to generate level 3 data.

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

          known_issues
          contributing
          design
          reference/index
          changelog

Getting help
============

If you would like to get in touch with someone who works on ``muse`` **for any reason**, we suggest opening an issue on the `muse GitHub issue tracker <https://github.com/LM-SAL/muse/issues>`__.
