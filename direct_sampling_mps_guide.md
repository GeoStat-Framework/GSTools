# Direct Sampling and Multiple-Point Geostatistics

Reading guide for Mariethoz, Renard, and Straubhaar (2010), "The Direct
Sampling method to perform multiple-point geostatistical simulations."

Project copy:
`/Users/lealroja/Documents/UFZ/MPS-Tools/GSTools/direct_sampling_mps_guide.md`

## Index

- [1. The Main Idea Of The Paper](#1-the-main-idea-of-the-paper)
- [2. Vocabulary Before The Algorithm](#2-vocabulary-before-the-algorithm)
- [3. Multiple-Point Geostatistics In Simple Terms](#3-multiple-point-geostatistics-in-simple-terms)
- [4. Multiple-Point Geostatistics More Formally](#4-multiple-point-geostatistics-more-formally)
- [5. What A ccdf Means](#5-what-a-ccdf-means)
- [6. What snesim Does](#6-what-snesim-does)
- [7. What Direct Sampling Does](#7-what-direct-sampling-does)
- [8. Figure 1 Explained Slowly](#8-figure-1-explained-slowly)
- [9. The DS Algorithm Step By Step](#9-the-ds-algorithm-step-by-step)
- [10. Parameters That Matter](#10-parameters-that-matter)
- [11. Common Misunderstandings](#11-common-misunderstandings)
- [12. If You Remember Only Five Things](#12-if-you-remember-only-five-things)
- [Sources](#sources)

## 1. The Main Idea Of The Paper

Yes: Direct Sampling, usually abbreviated as DS, is one method inside the
broader family of Multiple-Point Geostatistics, also called Multiple-Point
Statistics or MPS.

MPS is the general modeling idea. DS is one algorithm that implements that idea.
`snesim` is another important algorithm in the same family.

The paper is about making MPS easier and more flexible. Before DS, a common MPS
workflow was:

1. Take a training image.
2. Scan it before simulation.
3. Count many local spatial patterns.
4. Store those counts in a database or tree.
5. During simulation, use that stored database to decide what value to simulate
   next.

That is the logic used by `snesim`.

Mariethoz et al. propose a different approach:

1. Do not build the full database first.
2. When you need to simulate one unknown grid cell, look at its already known
   neighbors.
3. Search directly inside the training image for a similar neighborhood.
4. When a good match is found, copy the center value from that training-image
   location into the simulation.

That is DS.

This avoids storing all possible pattern probabilities. That matters because
the number of possible patterns can become enormous, especially in 3D, with many
facies, with continuous values, or with several variables at the same time.

The paper's main claim is not that DS invents a completely different statistical
goal. The goal is still MPS: reproduce realistic spatial patterns from a
training image while honoring available data. The novelty is how DS reaches that
goal.

## 2. Vocabulary Before The Algorithm

### Training Image, TI

A training image is an example image that contains the kinds of spatial patterns
you want your simulation to imitate.

In hydrogeology, the TI might represent geological facies: channels, sand
bodies, clay lenses, connected high-permeability zones, and so on. It is not
necessarily a map of the real site. It is more like a pattern library.

Mental model:

> The training image is the textbook of patterns. The simulation learns by
> copying patterns from this textbook.

### Simulation Grid, SG

The simulation grid is the grid you want to fill. At the start, most cells are
unknown. Some cells may already be known because you measured them.

Mental model:

> The simulation grid is the blank exercise sheet.

### Conditioning Data

Conditioning data are values that must be honored. For example, if a borehole
says that a certain location is sand, then the simulation should keep sand at
that location.

They are called "conditioning" data because they condition, constrain, or guide
the simulation.

### Realization

A realization is one possible simulated world. Repeating the simulation with
different random choices gives different realizations that should all look
compatible with the training image and conditioning data.

Mental model:

> A realization is one possible answer among many plausible answers.

### Neighborhood

For one unknown cell, the neighborhood is the set of nearby cells that are
already known or already simulated.

DS does not look at every cell in the whole grid when simulating one location.
It looks at a limited number of already informed neighbors.

### Lag Vector

A lag vector tells you where a neighbor is relative to the cell you want to
simulate.

If the unknown cell is at position `x`, and one known neighbor is one step to the
right and two steps up, the lag vector is `(1, 2)`.

The important part is that lag vectors are relative positions, not absolute
coordinates.

### Data Event

A data event is the list of values found at the neighbor positions.

If the lag vectors tell us where to look, the data event tells us what values we
find there.

Mental model:

> The lag vectors are the shape of the stencil. The data event is the pattern of
> values inside that stencil.

## 3. Multiple-Point Geostatistics In Simple Terms

Imagine you are trying to complete a partially missing picture.

If you only use two-point information, you mostly ask:

> How similar are two locations as a function of distance?

That is the usual variogram/covariance way of thinking. It is powerful, but it
often struggles with shapes. For example, it may know that nearby cells are
related, but it may not understand that a geological channel should stay long,
thin, curved, and connected.

MPS asks a richer question:

> What local arrangements of many points tend to occur together?

Instead of only looking at pairs, it looks at patterns involving several points
at once.

A simple analogy:

- Two-point geostatistics is like learning a language by checking pairs of
  letters.
- Multiple-point geostatistics is like learning by looking at short words or
  fragments of sentences.

For geological images, the equivalent of "realistic text" is realistic spatial
structure: connected channels, lenses, barriers, curving patterns, repeated
textures, or categorical arrangements.

## 4. Multiple-Point Geostatistics More Formally

In geostatistics, we often represent the unknown spatial variable as `Z(x)`.
That means:

> The value of variable `Z` at location `x`.

For example:

- `Z(x)` might be facies at location `x`: sand, clay, gravel.
- `Z(x)` might be hydraulic conductivity.
- `Z(x)` might be porosity.

When simulating one unknown location `x`, MPS looks at nearby known values:

```text
dn = {Z(x1), Z(x2), ..., Z(xn)}
```

This is the data event. It means:

> Here are the `n` neighboring values around the location we want to simulate.

MPS then asks:

```text
What is the probability distribution of Z(x), given this nearby pattern dn?
```

The paper writes this idea as a conditional cumulative distribution function:

```text
F(z, x, dn) = Prob{Z(x) <= z | dn}
```

Do not let the notation hide the simple idea. It means:

> Given the neighboring pattern I already see, what values are likely at the
> center?

The training image is used to estimate this: find similar surrounding patterns
in the TI, then look at their center values.

## 5. What A ccdf Means

`ccdf` means conditional cumulative distribution function.

Break the phrase apart:

- `distribution function`: describes probabilities of possible values.
- `cumulative`: gives probabilities of being less than or equal to a value.
- `conditional`: the probabilities depend on information you already know.

In this paper, the known information is the local data event around the unknown
node.

Simple example with categories:

Suppose you want to simulate the center cell, and the surrounding pattern is:

```text
left = sand
right = sand
top = sand
bottom = clay
```

You search the training image for places where that same surrounding pattern
occurs. Imagine you find 10 matches. In those 10 matches, the center value is:

- sand 8 times
- clay 2 times

Then the conditional distribution says:

- probability of sand is about 0.8
- probability of clay is about 0.2

The important point:

> The ccdf is the probability rule for the unknown center value, after looking
> at the known neighboring pattern.

Classical MPS algorithms try to estimate or store this probability rule.
DS avoids explicitly building it.

## 6. What snesim Does

`snesim` is one of the classic MPS algorithms. The name comes from "single
normal equation simulation." You do not need that name to understand the logic.

`snesim` works in two broad phases.

First, before the actual simulation, it scans the training image. It looks for
many possible data events and stores their statistics in a search tree. For each
local pattern, it records how often different center values occur.

Second, during simulation, it visits unknown cells one by one. At each cell, it:

1. Looks at the already known neighboring values.
2. Forms a data event.
3. Queries the search tree.
4. Gets a conditional probability distribution.
5. Draws a value from that distribution.
6. Moves to the next cell.

The advantage is speed during simulation: once the search tree exists,
probabilities can be retrieved quickly. The disadvantage is memory. The number
of possible local patterns grows very quickly. For example, with 4 facies and a
30-node template, the theoretical number of possible patterns is `4^30`.

Because of this, `snesim` often needs a limited template size. That can make it
harder to capture large structures unless extra strategies, such as multigrids,
are used.

Short version:

> `snesim` scans first, stores pattern probabilities, then simulates from the
> stored probability tree.

## 7. What Direct Sampling Does

DS removes the precomputed probability database.

Instead of saying:

> First I will store all pattern probabilities, then I will simulate.

DS says:

> When I need a value, I will search the training image directly for a similar
> local pattern and copy the center value from the first acceptable match.

At one simulation location `x`, DS does this:

1. Look at the already simulated or conditioning neighbors around `x`.
2. Record their relative positions. These are the lag vectors `L`.
3. Record their values. This is the data event `dn(x,L)`.
4. Search the training image for a location `y` where the same relative
   positions contain a similar pattern `dn(y,L)`.
5. If the match is good enough, copy the center value `Z(y)` into the simulation
   at `Z(x)`.

For categorical variables, "similar" often means the fraction of mismatching
nodes is small. For continuous variables, "similar" can mean the numeric
differences are small according to a distance formula.

There is an acceptance threshold, usually written as `t`. If the distance is
less than this threshold, the candidate is accepted. If no good candidate is
found after scanning a limited fraction of the TI, DS uses the best candidate
found so far.

Short version:

> DS does not explicitly compute the ccdf. It searches the TI directly in a way
> that is statistically equivalent to sampling from the TI-conditioned
> distribution.

## 8. Figure 1 Explained Slowly

Figure 1 in the paper shows the core of DS.

### Figure 1a: Define the data event in the simulation grid

The gray or question-mark cell is the unknown cell to simulate. Around it, some
cells are already known. In the figure, there are three informed neighbors.

The paper gives their lag vectors:

```text
L = {(1, 2), (2, 1), (-1, 1)}
```

This means:

- first neighbor: 1 cell right, 2 cells up from the unknown cell
- second neighbor: 2 cells right, 1 cell up
- third neighbor: 1 cell left, 1 cell up

These are relative positions, not global coordinates.

Now the paper writes:

```text
dn(x,L) = {0, 0, 1}
```

This simply means:

> Around the simulation-grid location `x`, if we look at the three positions
> described by `L`, the values we find are 0, 0, and 1.

In table form:

| Lag vector | Meaning | Value in SG |
| --- | --- | --- |
| `(1, 2)` | 1 right, 2 up | `0` |
| `(2, 1)` | 2 right, 1 up | `0` |
| `(-1, 1)` | 1 left, 1 up | `1` |

That table is all `dn(x,L) = {0, 0, 1}` means.

### Figure 1b: Define the search window in the training image

Now DS wants to search the training image for the same shape of neighborhood.
But not every training-image location can be used as a candidate center.

Why? If a candidate center is too close to the border of the TI, one lag
position may fall outside the image. For example, if one lag vector says "2
cells right," a candidate center too near the right edge is invalid.

That is what the dimensions `a`, `b`, `c`, and `d` are about.

They are not randomly selected.

They are the distances needed to keep the full data-event shape inside the
training image. They come from the minimum and maximum components of the lag
vectors.

They may be longer on one side than another because the current data event is
not symmetric. In this example, the known neighbors are mostly above and to the
right of the unknown node. So the required margin is different in different
directions.

Mental model:

> `a`, `b`, `c`, and `d` are safety margins. They define where a candidate
> center can be placed in the TI without the stencil falling outside the image.

### Figure 1c: Start scanning the training image

DS chooses a random starting location inside the valid search window. From
there, it scans the search window systematically.

At one candidate training-image center `y`, the paper gives:

```text
dn(y,L) = {1, 0, 1}
```

This uses the same lag vectors `L`, but now around the training-image candidate
location `y`.

In table form:

| Lag vector | Value in SG around `x` | Value in TI around `y` |
| --- | --- | --- |
| `(1, 2)` | `0` | `1` |
| `(2, 1)` | `0` | `0` |
| `(-1, 1)` | `1` | `1` |

So:

```text
dn(x,L) = {0, 0, 1}
dn(y,L) = {1, 0, 1}
```

Only the first value differs.

For a categorical variable, the simple mismatch distance is:

```text
number of mismatches / number of compared nodes
```

Here:

```text
1 mismatch / 3 nodes = 1/3
```

So the distance is `1/3`, or about `0.333`.

If the acceptance threshold `t` is greater than or equal to this value, this
candidate may be accepted. If it is stricter, DS keeps scanning.

### Figure 1d and 1e: Accept a match and copy the center value

When DS finds a candidate where the data event matches well enough, it copies
the center value from the TI candidate location into the unknown simulation
location.

Important:

> DS does not copy the neighbor values. It uses the neighbor values only to find
> a matching pattern. Then it copies the center value.

That center value becomes a newly simulated value in the SG. Later, it can be
used as a neighbor for other unknown locations.

## 9. The DS Algorithm Step By Step

Here is the algorithm in the order it happens.

### Step 1: Place hard or conditioning data

If you have measured data, put them into the simulation grid first. These values
are fixed information that the simulation should respect.

### Step 2: Choose a simulation path

The simulation path is the order in which unknown cells will be visited. It can
be random or follow another rule.

The path matters because after a cell is simulated, it becomes available as
conditioning information for later cells.

### Step 3: At one unknown node, find already informed neighbors

For the current unknown location `x`, find nearby cells that already have
values. These can be original conditioning data or values simulated earlier.

The paper usually keeps only up to `n` closest informed neighbors.

### Step 4: Convert neighbors into lag vectors

For each neighbor, compute its relative position from `x`.

If a neighbor is at `x_i`, the lag vector is:

```text
h_i = x_i - x
```

The set of all lag vectors is:

```text
L = {h_1, h_2, ..., h_n}
```

This set defines the shape of the local pattern.

### Step 5: Read the simulation-grid data event

Now read the values at those lag positions around `x`:

```text
dn(x,L) = {Z(x + h_1), Z(x + h_2), ..., Z(x + h_n)}
```

This is the pattern that DS wants to match in the training image.

### Step 6: Define the valid training-image search window

DS needs candidate centers `y` in the TI such that all positions `y + h_i` are
inside the TI.

This avoids trying to compare a pattern that goes outside the image boundary.

### Step 7: Start scanning from a random TI location

DS randomly chooses a starting candidate inside the valid search window. This
keeps the method stochastic, meaning repeated runs can produce different
realizations.

### Step 8: Compare candidate data events

For each candidate center `y`, DS reads:

```text
dn(y,L) = {Z(y + h_1), Z(y + h_2), ..., Z(y + h_n)}
```

Then it computes a distance:

```text
d{dn(x,L), dn(y,L)}
```

For categorical data, the distance can be the fraction of mismatches. For
continuous data, it can be a normalized numeric distance.

### Step 9: Accept the first good enough candidate

If the distance is below the threshold `t`, DS accepts the candidate.

If no candidate is good enough after scanning the allowed fraction of the TI,
DS accepts the candidate with the smallest distance found so far.

### Step 10: Copy the center value

Finally:

```text
Z(x) = Z(y)
```

The value at the candidate center in the training image is pasted into the
unknown cell in the simulation grid.

Then DS moves to the next node in the simulation path.

## 10. Parameters That Matter

### Training image quality

The TI is the source of the patterns. If the TI does not contain the type of
structure you want, DS cannot magically invent it in a reliable way.

Bad TI, bad pattern library.

### Number of neighbors, `n`

The parameter `n` controls how many already informed neighbors are used in the
data event.

Larger `n` means more context. This can improve pattern reproduction, but it can
also make exact matches harder to find.

Smaller `n` means easier matching, but less spatial structure is considered.

### Acceptance threshold, `t`

The threshold controls how strict DS is when accepting a match.

Small `t`:

- stricter matching
- better local pattern reproduction
- slower simulation
- harder for continuous variables, where exact matches are rare

Large `t`:

- easier matching
- faster simulation
- more tolerance for imperfect patterns
- possibly less realistic pattern reproduction

### Scan fraction

DS does not always scan the whole TI for every simulated node. A scan fraction
limits how much of the TI can be searched before accepting the best candidate
found so far.

Larger scan fraction means better chance of finding a good match, but more CPU
time.

### Distance function

The distance function defines what "similar" means.

For categorical facies, do not use ordinary numeric distance between category
codes. If facies are coded as 1, 2, and 3, that does not mean facies 1 is
"closer" to facies 2 than to facies 3. The codes are labels, not numbers with
physical order.

For continuous variables, numeric distances make more sense.

### Simulation path

The path affects which values are available as neighbors at each step. Random
paths are common because they avoid always growing the simulation in one fixed
direction.

## 11. Common Misunderstandings

### "DS is the same thing as MPS."

Not exactly. MPS is the general framework. DS is one algorithm in that framework.

A good analogy:

- MPS is "transportation."
- DS is "a bicycle."
- `snesim` is "a car."

Both DS and `snesim` are ways to do MPS, but they operate differently.

### "The training image is the true map."

Not necessarily. The TI is a source of plausible spatial patterns. It may be
based on geological interpretation, process modeling, an analog outcrop, remote
sensing, or another conceptual model.

### "The dimensions a, b, c, and d in Figure 1 are random."

No. They are determined by the shape of the current data event. They are margins
used to define where the stencil can fit inside the TI.

They are different lengths because the known neighbors around the unknown cell
are not evenly spread in all directions.

### "`dn(x,L)` contains coordinates."

No. `L` contains relative coordinates. `dn(x,L)` contains values.

In the Figure 1 example:

```text
L = {(1, 2), (2, 1), (-1, 1)}
```

means where to look.

```text
dn(x,L) = {0, 0, 1}
```

means what values were found there in the simulation grid.

### "`dn(y,L)` uses a different shape."

No. It uses the same lag vectors `L`. The only difference is that the same shape
is placed around a candidate location `y` in the training image.

The comparison is: same stencil shape, different center location.

### "DS copies the whole pattern."

In the basic algorithm described in this paper, DS copies only the center value
`Z(y)` into `Z(x)`. The surrounding pattern is used to decide whether `y` is a
good candidate.

Some other MPS algorithms paste patches, but basic DS is a pixel-based method.

### "The ccdf disappears in DS."

The concept does not disappear. DS avoids explicitly computing and storing it.

The paper's point is that direct sampling from the TI, conditional on the local
data event, can produce the same kind of result without building the ccdf table
first.

### "A lower threshold is always better."

Not always. A very strict threshold can make simulation slow or difficult,
especially with continuous variables. A threshold is a tradeoff between pattern
quality and computation time.

## 12. If You Remember Only Five Things

1. DS is one algorithm within Multiple-Point Geostatistics, not a separate
   field.
2. MPS uses training images to reproduce multi-cell spatial patterns, not only
   pairwise similarity.
3. `snesim` precomputes and stores pattern probabilities; DS searches the
   training image directly during simulation.
4. In Figure 1, `L` is the list of relative neighbor positions, while `dn(x,L)`
   and `dn(y,L)` are the values found at those positions.
5. DS accepts a training-image candidate when its surrounding pattern is close
   enough to the simulation-grid pattern, then copies the candidate center value.

## Sources

- Mariethoz, G., Renard, P., and Straubhaar, J. (2010). "The Direct Sampling
  method to perform multiple-point geostatistical simulations." Water Resources
  Research, 46, W11536. DOI:
  [10.1029/2008WR007621](https://doi.org/10.1029/2008WR007621). This guide is
  based primarily on the local PDF in `/Users/lealroja/Documents/UFZ/MPS-Tools/PDFs`.
- Meerschman, E., Pirot, G., Mariethoz, G., Straubhaar, J., Van Meirvenne, M.,
  and Renard, P. (2013). "A practical guide to performing multiple-point
  statistical simulations with the Direct Sampling algorithm." Computers &
  Geosciences, 52, 307-324. [UGent bibliography page](https://biblio.ugent.be/publication/3061447),
  DOI: [10.1016/j.cageo.2012.09.019](https://doi.org/10.1016/j.cageo.2012.09.019).
- Mariethoz, G. and Caers, J. (2014). Multiple-point Geostatistics:
  Stochastic Modeling with Training Images. [Wiley overview](https://www.wiley-vch.de/en/areas-interest/natural-sciences/earth-science-11es/geology-geophysics-11es1/multiple-point-geostatistics-978-1-118-66275-5).
- Optional learning aid from your colleague:
  [Claude public artifact](https://claude.ai/public/artifacts/25bac346-deea-43bb-8c3b-bc920d9fae35).
  I could not reliably inspect the artifact contents from this environment, so
  it is included as a supplementary visualization link, not as a source for the
  explanations above.
