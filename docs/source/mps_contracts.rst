.. _mps-contracts:

Direct Sampling controls and reproducibility
=================================================

Direct Sampling (DS) uses the GSTools-Core Rust engine. GSTools installs Core
as a required dependency; DS additionally checks the native MPS interface and
required exports before running. A package version alone does not identify the
source of a development wheel. Record both repository revisions and the wheel
hash when comparing development builds.

Construct training images and zones with ``gs.mps.TrainingImage`` and
``gs.mps.Zone``. ``gs.MPSModel`` and ``gs.DirectSampling`` are available at the
top level.

Large grids
-----------

Large structured grids require a finite search radius when the unbounded
lag table would exceed five million entries. Set the radius for each variable
in the main and cleanup configurations. This example simulates a 100 × 100 ×
100 grid with one-cell search radius in index units::

    import numpy as np
    import gstools as gs

    image = gs.mps.TrainingImage(np.ones((3, 3, 3)), n_neighbors=1, max_radius=1)
    ds = gs.DirectSampling(gs.MPSModel(image, scan_fraction=0.1), seed=42)
    field = ds([np.arange(100)] * 3, num_threads=4, store=False)

Candidate selection
-------------------

``threshold=0.0`` is the default best-candidate (DSBC) mode. ``None``, integer
zero, and negative zero are equivalent inputs. DS returns the lowest-distance
selectable candidate examined within the scan budget; it can stop at an exact
zero distance because no smaller nonnegative distance exists. This does not
require an exact match and does not imply a full-TI search. Ties retain the first
minimum encountered.

For a finite positive threshold, the first selectable candidate with distance
**strictly below** the threshold is accepted. If none qualifies, DS uses the
best visited candidate. Negative and nonfinite thresholds are rejected. Values
above one accept the first selectable match for distances bounded by one. Lp
distances can exceed one when conditioning lies outside the training-image
range. The raw DS field preserves hard conditioning at every threshold.

For a nonempty valid candidate window, the budget is
``max(1, min(window_size, floor(scan_fraction * selected_TI_size)))``. The
selected TI is the zone TI at a zoned node. Cleanup uses the effective scan
fraction after division by ``post_processing_factor``. The optimum is over the
examined selectable candidates, not all TI locations. Empty events and searches
without selectable candidates use the engine's unconditional fallback; these
are distinct from malformed native input, which raises a validation error.
With missing TI values, fallback draws from centers finite in all variables.

The controls address different orders and budgets:

* ``path`` sets the simulation-grid visit order.
* ``scan_path`` sets the TI candidate order at each node. Sequential scanning
  wraps raster order from a node-local random start; random scanning samples
  distinct positions without replacement.
* ``post_processing_path`` sets the cleanup revisit order.
* ``scan_fraction`` sets the candidate budget relative to TI size, capped by
  the valid window.
* ``threshold`` sets approximate-match early acceptance.

Coordinates and transformations
-------------------------------

Neighborhood offsets and ``max_radius`` use **grid-index units**. Physical axes
are used for coordinates, conditioning-point snapping, and coordinate-dependent
callables. Unequal axis spacing and nonuniform axes do not turn the neighbor
search into a physical-distance search. Use finite axes and account explicitly
for index-space geometry when choosing radii and transformations.

Rotation angles are in **radians**, in GSTools angle ordering. Scale factors
are positive and dimensionless, one per axis. They transform simulation-grid
lags into TI lags; they do not rotate or resample the TI data. The inverse map
is ``M = diag(1 / scale) @ matrix_derotate(dim, angles)``; row-vector lags use
``lags_ti = lags_sg @ M.T`` before integer-grid rounding. For example, in 2D,
``rotation=np.pi/2`` with ``scale=[2, 1]`` applies a quarter-turn and a factor-two
stretch of simulated structures along the first local axis relative to the TI.
Exact identity settings preserve the untransformed path.

Distances and penalties
-----------------------

Variation distance compares deviations from local event means. Both means use
the same finite comparison support. For each supported pair, the centered
residual is divided by ``2*d_max``; the weighted Lp aggregate is clipped at one.
Masked comparisons renormalize weights on their valid support. The pasted
center is shifted by the difference of the event means on that same support.
Simulated values can therefore lie outside the original TI range. Requested
variation neighborhoods need at least two points, but the realized common
support can be smaller.

Thresholds apply to this implemented normalization. Dividing a centered
residual by ``2*d_max`` rather than ``d_max`` reduces its distance and makes a
fixed positive threshold easier to satisfy before clipping. Thresholds are not
automatically transferable between those normalizations or between variation
and absolute-value metrics.

For categorical variables, a penalty matrix replaces plain mismatch costs with
``T[simulation_category, TI_category]``. Asymmetric costs are supported, the
diagonal is zero, and off-diagonal entries lie in ``(0, 1]``. These costs compare
corresponding event values. They do not prohibit adjacent facies pairs or
directly impose a pairwise transition energy on the simulated grid. Spatial
transition statistics can change indirectly and should be measured over an
ensemble.

Cleanup and zones
------------------

**PostProcessing** groups the number of passes, search effort, revisit order,
sparse variable overrides, and the cleanup mask. For example, for a training
image with a named variable "facies"::

    cleanup = gs.mps.PostProcessing(2, factor=2, path="same",
                                   overrides={"facies": {"n_neighbors": 12}})
    model = gs.MPSModel(ti, post_processing=cleanup)

The effective neighbor count is max(1, round(n / factor)) with ties rounded
to even. Variation metrics require at least two requested neighbors. The
factor also divides the scan fraction, with the candidate budget capped by
the window. Cleanup revisits active nodes, restricted by its own optional
grid-shaped mask. Partially conditioned components remain fixed. Each pass
uses values updated earlier in that pass.

For continuous variables, the separate call-time option ``post_process=False`` disables
the Field mean/normalizer/trend pipeline for access to raw values. Categorical
variables retain their labels. Multivariate calls return a dictionary keyed by
variable name and, when saved, store the fields under those names.

Zones select TI data, shape, value ranges, and missingness by domain. Search
settings come from the effective primary or cleanup configuration, not the
zone TI's unused search settings. Events can include neighbors across zone
boundaries; this permits interaction without guaranteeing a seamless boundary.
For multivariate simulation, an unadjusted collocated vector copied without
retained partial conditioning comes from one TI location. That property does
not guarantee exact finite-realization marginal frequencies, connectivity, or
multipoint distributions.

Random streams and threads
--------------------------

DS uses the opt-in ``SeedSequence``/``MT19937`` contract
``ds-seedsequence-mt19937-v1``. SeedSequences spawn child streams without
compressing them to the former 16-bit seed space. This intentionally changes
seeded DS realizations. Default seeding for other GSTools features is unchanged.
NumPy spawning provides a very high probability of independent streams, not a
proof that every finite output differs for every seed.

Constructor ``seed=None`` or default NaN creates an entropy-seeded master.
At call time, an omitted ``seed`` or NaN continues the master sequence; an
integer resets it reproducibly; ``seed=None`` creates a fresh entropy-seeded
master. Omitted, NaN, or None ``path_seed`` and ``node_seed`` derive their streams
from the master; explicit integers select repeatable role streams independent
of unrelated master changes. Equal explicit role seeds can initialize equal
streams. A fixed path stream fixes path controls; changing node order can still
change the final field even when the node stream is fixed.

Thread selection uses call-time ``num_threads``, then instance
``num_threads``, then ``gstools.config.NUM_THREADS``. The current unset/falsy
fallback is **one thread**, not all cores. Paths shorter than 512 nodes run
serially; larger dependency graphs may still offer little useful concurrency.
Within a fixed supported build and dependency environment, identical seed
controls and inputs are intended to produce identical fields at different DS
thread counts. This does not promise bitwise identity across operating systems,
compiler/dependency versions, or algorithm releases. Record NumPy, source
revisions, stream contract, controls, scan mode, and effective threads with
reproducibility results.

Scientific checks
------------------

A plausible plot or passing deterministic tests does not establish ensemble
quality. Validate the statistics needed for the application, for example
facies proportions, face-connected connectivity, spatial correlations, and
cross-variable distributions. Use realizations as the independent units for
Monte Carlo uncertainty, rather than treating spatially dependent pixels as
independent samples. Variation with shifted conditioning needs centered or
detrended targets, rather than raw TI marginals alone. Example contact counts
and cleanup effects describe their particular realizations.

Numeric input limits
---------------------

The DS engine stores values as float64. Integer-typed inputs are checked for
exact float64 representability before packing. Integers in ``[-2**53, 2**53]``
are representable; some larger values, such as powers of two, are also exact.
Other large labels or values raise an actionable error: recode categorical
labels to small integers, or center/rescale continuous data before conversion.
For example, ``2**60`` is exact but ``2**60 + 3`` is not. This check cannot
recover distinctions already lost in an input converted to floating point.

``n_neighbors`` and ``post_processing`` accept integer types without converting
them through float. Integral floats and booleans remain accepted for
compatibility; fractional values, strings, and nonfinite numbers are rejected.
Neighbor counts must be positive; cleanup pass counts may be zero.


Active simulation domains
-------------------------

``MPSModel(active_mask=mask)`` accepts a boolean array with exactly the grid
shape. True cells are active. False cells are excluded from the main simulation
path, all neighbor events and cleanup paths/dependency graphs, and remain NaN
in every output variable. They are not filled and subsequently erased. Hard
conditions outside the active mask are rejected before control RNG streams are
acquired. ``post_processing_mask`` narrows cleanup targets only; it does not
replace an active simulation mask. Zone TIs and partial multivariate conditions
use the same active domain.

``None`` preserves the full rectangular domain. An all-True mask is tested to
match ``None`` exactly within a fixed build; an all-False mask yields NaN output.
An explicit path may contain inactive extras (which are dropped) but must cover
every unknown active node. The mask excludes contributors, not geometric lines
between nodes: two active cells across a hole may remain neighbors if within
radius. Disconnected components farther apart than the search radius do not
exchange event values. Grid-index distance and transformed-coordinate units
remain unchanged.

The paired Python/Core implementation requires MPS native ABI 8. Development
Core versions sharing the package version are distinguished by ABI and binary
hash, not by version number alone. When benchmarking, report the active mask
and active cell count so readers can identify the workload.
