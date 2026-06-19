Multiple Point Statistics
=========================

Two-point geostatistics (covariance models, kriging, SRFs) describes spatial
structure through pairs of points and a variogram. This is powerful, but it
cannot reproduce *curvilinear* or *connected* features such as meandering
channels, fractures, or other patterns that depend on the joint configuration
of many points at once.

**Multiple Point Statistics (MPS)** addresses this by learning patterns
directly from a **training image (TI)** — an example image deemed
representative of the spatial structure to simulate. Instead of fitting a
variogram, MPS borrows whole patterns from the TI.

GSTools provides the **Direct Sampling (DS)** algorithm
(`Mariethoz et al., 2010 <https://doi.org/10.1029/2008WR007621>`_), together
with the **Direct Sampling Best Candidate (DSBC)** parametrization
(`Juda et al., 2022 <https://doi.org/10.1016/j.acags.2022.100091>`_), through
two classes:

* :any:`TrainingImage` — the MPS model: the training image plus the distance
  used to compare patterns (the analogue of a :any:`CovModel`).
* :any:`DirectSampling` — the generator that produces realizations on a
  structured grid (the analogue of :any:`SRF`).

The core idea: to fill each grid cell, DS looks at the values already present
around it (its *data event*), scans the training image for a location whose
surroundings look similar enough, and copies that cell's value over.
"Similar enough" is decided by a **distance** between the two surroundings,
controlled by three parameters — the number of neighbours ``n``, the scan
fraction ``f``, and the acceptance threshold ``t`` (with ``t = 0`` giving the
recommended DSBC mode).

The following tutorials build up from a minimal unconditional simulation to
conditioning and continuous variables.

Examples
--------
