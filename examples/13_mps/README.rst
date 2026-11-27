Multiple Point Statistics
=========================

Two-point geostatistics describes spatial structure with pairs of points:
covariance models, variograms, kriging, and spatial random fields. That is
often enough, but it is not a natural language for connected or curvilinear
patterns such as channels, fractures, lenses, or object-like facies bodies.

**Multiple Point Statistics (MPS)** uses a **training image (TI)** instead: a
N-dimensional structured array containing patterns considered representative
of the field to simulate. The TI may be categorical, such as integer facies
codes, or continuous, such as permeability, porosity, elevation, or another
property. The examples in this gallery focus on 2D and 3D grids.

Use MPS when the geometry of the pattern matters more than a variogram can
express. If a covariance model already describes the field well, :any:`SRF`
is usually simpler and faster; for categorical fields generated from
continuous Gaussian fields, :any:`PGS` may also be a better first choice.

Direct Sampling in GSTools
--------------------------

GSTools implements the **Direct Sampling (DS)** algorithm
(`Mariethoz et al., 2010 <https://doi.org/10.1029/2008WR007621>`_) with the
**Direct Sampling Best Candidate (DSBC)** parametrization recommended by
`Juda et al. (2022) <https://doi.org/10.1016/j.acags.2022.100091>`_.
The API follows the usual GSTools model/generator pattern:

* :any:`TrainingImage` stores one TI, or a list of named :any:`Variable`
  objects for multivariate simulation.
* :any:`MPSModel` stores the Direct Sampling parameters.
* :any:`DirectSampling` generates a realization on a structured grid.

At each unsimulated grid node, DS gathers the already known neighboring values
around that node. This neighborhood is the *data event*. DS then searches the
TI for candidate neighborhoods with the same relative cell positions and
similar values; when a good candidate is found, the value at the candidate
center is copied into the simulation grid. In an unconditional simulation,
these known neighbors are values simulated earlier in the random path; hard
data simply add measured values that must be honored.

Three parameters control the quality and runtime tradeoff:

* ``n_neighbors`` (``n``): the maximum number of known neighbors in each data
  event. Larger values can capture richer patterns but cost more.
* ``scan_fraction`` (``f``): the fraction of the TI search examined for each
  simulated node, measured relative to selected TI size and capped by the valid
  candidate window. Larger values search harder but run slower.
* ``threshold`` (``t``): an early-acceptance distance. Start with
  ``threshold=None`` (DSBC), which takes the best
  selectable candidate within its budget; exact zero-distance matches can
  stop early. Positive thresholds accept strictly below the threshold, falling
  back to the best visited candidate if none qualifies. Tune
  positive thresholds only when this is not sufficient.

The gallery is ordered as a learning path. Start with **A first Direct Sampling
simulation**, then add field measurements in **Conditioning to hard data**, and
switch from facies labels to continuous values in **Continuous variables and
distance metrics**.

The next pages move from small synthetic images to the bundled Strebelle
channel TI stored in ``input/strebelle_channel_ti.npz``. **A real training
image: the Strebelle channels** repeats the conditional workflow on a classic
MPS benchmark, and **Direct Sampling parameters: n_neighbors, scan_fraction,
threshold** then isolates the three quality/speed knobs and the DSBC-versus-DS
choice on that image. **Nonstationary geometry with the Strebelle channels**
introduces rotation and scale. Next,
:ref:`Herten outcrop: multi-facies simulation with a rotation gradient
<sphx_glr_examples_13_mps_07_herten_nonstationary.py>` compares stationary and
rotated search geometry on one categorical variable with eight facies in the
selected crop. Its default offline preview displays precomputed results; the
page documents explicit regeneration. Continue with
:ref:`Radial nonstationarity
<sphx_glr_examples_13_mps_08_channel_strebelle_radial.py>` for radial rotation
and spatial scaling before introducing multiple variables.

The remaining 2D examples broaden the data model. **Multivariate channels with
a context variable** follows a published example and uses an exhaustive
continuous context field to guide a categorical channel simulation. It
simulates only the channel variable; **Bivariate joint simulation from a
channel TI**, later in the gallery, jointly simulates both variables and
provides a second paper-based reproducibility example. **Parallel execution and
reproducibility** then shows that realizations are bit-identical across thread
counts within a fixed build and dependency environment, and how the ``seed`` /
``path_seed`` / ``node_seed`` controls work.
**Variation distance for continuous conditioning** then returns to continuous
conditioning and shows why comparing local variation can be useful when hard
data have a different absolute level than the TI. **Continuous conditioning
with a bundled texture** uses the
GAIA-UNIL ``stone`` training image and compares the simulated and training-image
histograms. This follows a published workflow. Its saved
figure is displayed by default.

Finally, the 3D examples change the dimensionality rather than the variable
type. **Large 3D multi-facies channel simulation** uses a five-facies GAIA-UNIL
Fluvsim crop with sparse conditioning points and paper-style validation plots
for facies proportions and two-point correlation, and **HESS Roussillon aquifer
reproduction** closes the gallery with an end-to-end real-world reproduction
combining a nonstationary training image, a 3D trend, rotation maps, and hard
conditioning data. Both display saved figures by default, with expensive
regeneration kept explicit. Regeneration of the Fluvsim example is expensive
and hardware-dependent; its page gives the command to regenerate the figures.

A final group returns to 2D to show the algorithm controls added on top of the
basic model. **Post-processing: cleanup passes** re-opens an already-simulated
field against the current neighborhood to revisit local matches
(``post_processing=PostProcessing(...)``). **Zoned simulation** scans a different training
image per region of the grid with :any:`Zone`. **Categorical penalty matrix**
replaces the hard 0/1 facies mismatch with soft, possibly ordinal, costs on a
:any:`Variable`. **TI scan order** contrasts the default sequential window scan
with the ``scan_path="random"`` extension. Each one is deterministic for a fixed
seed within a fixed build/dependency environment, including across thread
counts. These illustrative fields do not establish ensemble statistical fidelity.

Examples
--------
