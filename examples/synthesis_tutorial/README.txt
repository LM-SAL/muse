******************
Synthesis Tutorial
******************

The forward-modeling (synthesis) workflow for MUSE spectroscopic observations:

1. **VDEM** (velocity-differential emission measure): the emission measure as a function of velocity, temperature and position.
2. **Responses**: how the instrument responds to emission at each wavelength, velocity and temperature.
3. **Synthesis**: the VDEM combined with the responses gives synthetic spectra.
4. **Analysis**: the spectral moments of the synthetic spectra.

The tutorials save each dataset as a single ``.zarr.zip`` file in
``examples/synthesis_tutorial/artifacts``. Set ``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR``
to share a different output directory across runs. Each tutorial uses the local
dataset if available and otherwise downloads the published copy, so you can start at any step.

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Needs
     - Makes
     - Used by
   * - 01 Create a VDEM
     - PlasmaCalcs; downloads a 3 GB MURaM snapshot
     - the example VDEM and the emission-measure-weighted mean electron density of its bins
     - 02, 05, 08, 09
   * - 02 VDEM to MUSE's FOV
     - the example VDEM
     - the VDEM on MUSE's slits and raster steps
     -
   * - 03 CHIANTI line lists
     - ``muse[chianti]`` and a CHIANTI database in ``XUVTOP``
     - the line lists
     - 04, 07, 09
   * - 04 SG responses
     - the line lists; about 18.5 GB of memory at its peak
     - the 108, 171 and 284 Å SG responses
     - 05
   * - 05 Synthesize an observation
     - the example VDEM and the SG responses; about 13.7 GB of memory at its peak
     - the synthetic spectra
     - 06
   * - 06 Analyze the spectra
     - the synthetic spectra
     - their spectral moments
     -
   * - 07 Non-thermal responses
     - the 171 Å line list
     - a 171 Å response with non-thermal broadening
     -
   * - 08 CI response
     - the example VDEM and a 195 Å line list
     - a CI response and a CI image
     -
   * - 09 Synthesize with density
     - the example VDEM, its emission-measure-weighted density and the 171 Å line lists
     - a 171 Å image at the emission-measure-weighted density of each VDEM bin
     -
