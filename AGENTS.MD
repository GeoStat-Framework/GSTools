# Agent Context: Direct Sampling / MPS Discussion

This file summarizes the conceptual discussion with the user so a future agent
can continue without losing context. Update this file whenever answering a new
substantive question about this topic.

## Project Context

- Current project/workspace path:
  `/Users/lealroja/Documents/UFZ/MPS-Tools/GSTools`
- Main Markdown guide created for the user:
  `/Users/lealroja/Documents/UFZ/MPS-Tools/GSTools/direct_sampling_mps_guide.md`
- Source paper PDF discussed:
  `/Users/lealroja/Documents/UFZ/MPS-Tools/PDFs/Mariethoz et al. - 2010 - The Direct Sampling method to perform multiple‐point geostatistical simulations 1 (1).pdf`
- User also references an interactive DS page:
  `https://claude.ai/public/artifacts/25bac346-deea-43bb-8c3b-bc920d9fae35`

## Main Concepts Already Explained

- MPS means Multiple-Point Statistics or Multiple-Point Geostatistics.
- MPS is the broader framework; Direct Sampling (DS) is one algorithm inside
  MPS. `snesim` is another MPS algorithm.
- A training image (TI) is an example pattern library. The goal is not to
  copy-paste the TI exactly, but to generate new plausible spatial fields that
  reproduce TI-style spatial structures and honor measured data.
- The simulation grid (SG) is the domain to fill.
- Conditioning data are measured/known values placed in the SG, such as
  borehole facies, hydraulic conductivity, porosity, or geophysical constraints.
- A realization is one possible simulated spatial field. In practice DS is run
  many times to create an ensemble of realizations.
- For groundwater flow, typical workflow:
  1. Choose/build TI(s).
  2. Add conditioning data.
  3. Run DS many times.
  4. Convert facies to hydraulic properties if needed.
  5. Run groundwater flow/transport model on each realization.
  6. Summarize output uncertainty, not just average the geology maps.
- More than one TI can be used:
  - separate ensembles per geological concept,
  - combine compatible TIs,
  - use different TIs by zones,
  - weight scenarios by plausibility.

## Lag Vector Notes

- A lag vector is a relative displacement from the current node `x` to an
  informed neighbor `x_i`.
- Math:
  `h_i = x_i - x`
- The set of lag vectors is:
  `L = {h_1, h_2, ..., h_n}`
- The neighborhood positions are:
  `N(x,L) = {x + h_1, x + h_2, ..., x + h_n}`
- The data event is the list of values found at those relative positions:
  `dn(x,L) = {Z(x+h_1), Z(x+h_2), ..., Z(x+h_n)}`
- In the paper Figure 1 example:
  `L = {(1, 2), (2, 1), (-1, 1)}`
  and
  `dn(x,L) = {0, 0, 1}`
- The user asked why `h_3` is `(-1, 1)` instead of `(-1, -1)`. Explain using
  image/grid coordinates: the second coordinate may increase downward. Thus
  `(-1, 1)` can mean one cell left and one cell down in image coordinates.

## Distance / Threshold Notes

- For categorical variables, the paper defines mismatch distance as:
  `d = number of mismatching nodes / number of neighbors`
- More formally:
  `d{dn(x),dn(y)} = (1/n) * sum(alpha_i)`,
  where `alpha_i = 0` for a match and `1` for a mismatch.
- Example:
  `dn(x,L) = {0,0,1}` and `dn(y,L) = {1,0,1}` have one mismatch out of three,
  so `d = 1/3`.
- `t` is the acceptance threshold.
  - `t = 0`: exact match only.
  - small `t`: strict, slower, better local pattern reproduction.
  - large `t`: looser, faster, possibly less faithful to TI patterns.
- The paper mentions normalized distances in `[0,1]` and gives a rough sense
  that `0.05` is low/strict and `0.5` is high/loose.
- In the interactive page screenshot, `n = 8`, `t = 0`, `f = 0.50`, and best
  distance `0.125`. Since `0.125 = 1/8`, this means one mismatch among eight
  neighbors. Because `t = 0`, no exact match was found before the scan limit, so
  DS accepted the best candidate found after the allowed scans.
- Equation 2 in the paper is the unweighted categorical distance. Every neighbor
  has equal importance.
- Equation 3 is the weighted categorical distance. It still counts matches and
  mismatches, but mismatches can matter more or less depending on the distance
  from the central node. With the paper's examples using `d = 0`, all weights
  become equal, so equation 3 reduces to equation 2.
- When explaining equation 3, emphasize that it is not a new kind of pattern
  comparison. It is equation 2 plus weights: `distance = weighted mismatches /
  total weights`. The exponent parameter controls whether closer neighbors are
  more important. If the exponent is zero, all weights are one. The exponent is
  not limited to 0 or 1; it is a tuning parameter, usually chosen as a
  nonnegative real number. Larger values make close neighbors dominate more.
- Equations 4 and 5 are for continuous variables. They define a normalized
  weighted Euclidean distance between two data events. Instead of counting exact
  matches/mismatches, they measure numerical differences value by value,
  normalize by the total range of the TI (`dmax`), and optionally weight
  neighbor importance by distance from the center. The result is also in
  `[0,1]`, making it comparable with threshold `t`. In the basic paper
  definition, `dmax = max(TI values) - min(TI values)` is a global normalization
  constant computed from the TI, so it is calculated once per variable/TI unless
  using nonstationary/local normalization.
- Equation 6 is the suggested normalized Manhattan distance for continuous
  variables: average absolute difference divided by `dmax`. Compared with
  Euclidean distance, Manhattan distance penalizes large individual differences
  less strongly and is often more robust/simple. Euclidean is stricter about
  outlier mismatches because it squares differences.

## Data Event Radius Notes

- At each node `x`, DS selects up to `n` closest already informed nodes in the
  SG. These nodes can be conditioning data or previously simulated nodes.
- The data event radius is the distance from `x` to the farthest selected
  neighbor.
- Early in the simulation, few SG nodes are informed, so the `n` closest
  informed nodes can be far away. The data event radius is large.
- Later in the simulation, many nearby nodes have already been simulated, so the
  `n` closest informed nodes are closer. The data event radius naturally
  decreases.
- This natural shrinkage is why DS can capture larger-scale patterns early and
  smaller/local patterns later, somewhat replacing the multigrid idea used in
  `snesim`.

## Section 4 Continuous Variable Notes

- Section 4 shows why DS matters for continuous properties such as hydraulic
  conductivity, storativity, and porosity.
- Older MPS workflows often simulated categorical facies first, then assigned
  continuous properties to those facies in a second step. DS can directly
  simulate continuous fields with connected patterns, avoiding that mandatory
  two-step workflow.
- The example uses a continuous TI with connected low values. DS uses distance
  equation 4, 100 conditioning data, a large data event (`n = 80`), and a low
  threshold (`t = 0.01`).
- The 100 conditioning data in Section 4 are measured/fixed values placed in
  the SG. They are not the same as `n`. `n = 80` is the maximum number of
  already informed neighbors used in the data event for one simulated node.
- Important result: DS reproduces the TI's connectivity patterns and histogram
  while honoring conditioning data. This is relevant for groundwater because
  flow/transport behavior depends strongly on connected high/low hydraulic
  property zones.
- The 100 conditioning data in Section 4 are measured/fixed values placed in
  the SG. They are not the same as `n`. `n = 80` is the maximum number of
  already informed neighbors used in the data event for one simulated node.

## Section 5 Multivariate Case Notes

- Section 5 extends DS from one variable to several variables at once, e.g.
  facies plus resistivity, or geology plus geophysics.
- A multivariate TI is a "vector image": every location has a vector of values
  such as `(facies, resistivity)`.
- Each variable can have its own neighborhood, number of neighbors `n_k`, lag
  vectors `L_k`, data event, and distance function.
- The joint data event is the collection of all variable-specific data events.
- Equation 7 is the multivariate ccdf: the probability of variable `Z_k(x)` is
  conditioned on the data events from all variables, not only its own variable.
- Equation 8 combines individual variable distances into one joint distance:
  weighted average of distances, with user weights `w_k`, `sum(w_k)=1`, and
  `w_k >= 0`.
- Important use case: if a secondary variable is known everywhere, such as
  geophysics/resistivity, DS can use its spatial pattern to guide simulation of
  a primary variable, such as facies, even when direct primary measurements are
  sparse or absent.

## Section 6 Nonstationarity Notes

- In Section 6, nonstationarity mainly means spatial nonstationarity: the
  statistical behavior or pattern rules change with location in space. It does
  not primarily mean that hydraulic conductivity changes through time.
- Stationary means the same pattern-generating rules are assumed everywhere:
  same mean/proportions, same variability, same orientations, same connectivity
  style, etc. Nonstationary means at least one of these changes across the
  domain.
- Section 6 gives several DS ways to handle nonstationarity:
  1. use different TIs or TI zones for different simulation zones,
  2. use a custom distance adapted to nonstationary behavior,
  3. transform data events, e.g. rotation or affinity/scaling,
  4. add a secondary variable that describes where patterns should change.
- Equation 9 is a variation-based distance for continuous variables. It compares
  deviations from each data event's own local mean, not raw values. This lets a
  zero-mean TI match conditioning data with a much higher local mean if the local
  ups/downs around the mean have similar shape.
- In Figure 6, the TI is stationary with mean near 0 and values around -3.5 to
  4, while conditioning data are around 100-111 and nonstationary. Equation 9
  allows DS to borrow spatial variation/texture from the TI but shift simulated
  values to the local mean of the SG data event.

## Search Window Clarification

- The search window in the paper is a geometric valid-candidate region inside
  the TI.
- It is calculated from the current data-event stencil, i.e. from the selected
  neighborhood points around the current SG node and their lag vectors.
- In Figure 1, `a`, `b`, `c`, and `d` are border margins derived from the
  minimum and maximum horizontal/vertical components of the lag vectors.
- They are not random and they are not the same as scan fraction `f`.
- They define where a candidate center `y` can be placed in the TI so that all
  positions `y + h_i` remain inside the TI.
- If a candidate center is too close to the TI border, part of the stencil would
  fall outside the TI, so that candidate is invalid.
- Mental model: the data event is a stencil; the search window is the area where
  that stencil can slide without hanging outside the TI.
- The interactive page has a yellow rectangle in the TI. It may visualize either
  the valid search window or an implementation-specific scanned/display region.
  Treat it cautiously because it was developed by a colleague and may simplify
  or visualize the algorithm differently from the paper.
- `f` on the interactive page is the scan fraction:
  it controls what fraction of valid candidate locations are checked before DS
  falls back to the best candidate found. `f = 0.5` means about 50% of valid
  candidates/search-window locations, not necessarily exactly 50% of the whole
  TI.

## Current User Confusion To Address Carefully

- The user is still working through how exactly the search window is calculated.
- They correctly suspect it is based on selected neighborhood points and their
  distances/lag vectors.
- Future explanations should clearly separate:
  1. selected neighbors in SG,
  2. lag vectors/stencil shape,
  3. margins `a,b,c,d`,
  4. valid search window in TI,
  5. scan fraction `f`,
  6. match distance and threshold `t`.
- The user is now asking about the definition of the data event, why its radius
  decreases over time, and how to understand categorical distance equations 2
  and 3, plus continuous-variable equations 4 and 5.
