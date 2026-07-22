******************
Synthesis Tutorial
******************

This tutorial demonstrates the forward modeling (synthesis) workflow for MUSE spectroscopic observations.

The method involves:

1. **VDEM (Velocity-Differential Emission Measure)**: Contains the physical properties of the solar atmosphere (temperature, velocity, spatial structure)
2. **Response Functions**: Instrument-specific functions describing how the telescope and spectrograph respond to emission at different wavelengths and temperatures
3. **Synthesis**: Combining VDEM with response functions to produce synthetic spectra
4. **Analysis**: Calculating spectral moments and comparing with observations

This approach can be adapted to other instruments (e.g., AIA, EIS, EUVST) by using their respective response functions.

These examples are skipped during formal documentation builds because the full workflow exceeds Read the Docs' 7 GB memory limit.

Paths and caches
================

Downloaded inputs use Pooch's standard ``muse`` cache.
Set ``XDG_CACHE_HOME`` before running the tutorials to relocate it.
Examples that generate line lists, responses, or spectra write to ``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR`` when it is set, otherwise they use ``examples/synthesis_tutorial/artifacts``.
