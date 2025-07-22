"""
PGS through decision trees
--------------------------

In typical PGS workflows, the lithotype is defined through a spatial rule. A
more flexible approach can be taken such that the lithotype is represented as a
decision tree. More specifically, this is given as a binary tree, where each
node is a decision based on the values of the spatial random
fields [Sektnan et al., 2024](https://doi.org/10.1007/s11004-024-10162-5). The
leaf nodes are then assigned a discrete value which is given to the cell that
is being evaluated. Here, a simple example is provided showing how to use the
tree based approach in conducting plurigaussian simulation.

As in the previous examples, we define the simulation domain and generate the
necessary spatial random fields.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2

# no. of cells in both dimensions
N = [150, 150]

x = np.arange(N[0])
y = np.arange(N[1])

model = gs.Gaussian(dim=dim, var=1, len_scale=10)
srf = gs.SRF(model)
field1 = srf.structured([x, y], seed=215253419)
field2 = srf.structured([x, y], seed=192534221)

###############################################################################
# Decisions within the tree are carried out through user defined functions.
# In this way, the lithotype is defined in a continuous space, not relying on
# discretization. In this example we will use an ellipse as a decision boundary.
# The function accepts a data dictionary, which contains the values of the
# spatial random fields, and the parameters of the ellipse.


def ellipse(data, key1, key2, c1, c2, s1, s2, angle=0):
    x, y = data[key1] - c1, data[key2] - c2

    if angle:
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        x, y = x * c + y * s, -x * s + y * c

    return (x / s1) ** 2 + (y / s2) ** 2 <= 1


###############################################################################
# The decision tree is defined as a dictionary, where each node is a dictionary
# itself. The root node is the first decision, which branches into two nodes,
# one for each possible outcome. The leaf nodes are the final decisions, which
# assign a discrete value to the given cell. The `func` key in each decision
# node contains the function to be called, and the `args` key contains the
# arguments to be passed to the function. These arguments must match the
# parameters of the function. The `yes_branch` and `no_branch` keys contain the
# names of the nodes to follow based on the outcome of the decision. `root`
# must be given, but all other node names are arbitrary.
#
# Here, we define a simple decision tree with two phases. The first node
# checks if the point is inside an ellipse, and if it is, it assigns the value
# 1 to the lithotype. If it is not, it assigns the value 0. The parameters
# of the ellipse are defined in the `args` dictionary. The keys `key1` and
# `key2` refer to the spatial random fields, which are used to define the
# ellipse. In the algorithm, the fields are indexed as `Z1`, `Z2`, etc.,
# depending on the order in which they are passed to the PGS object. In this
# case, `Z1` refers to `field1` and `Z2` refers to `field2`. The parameters
# `c1`, `c2`, `s1`, `s2`, and `angle` define the center, scale, and rotation of
# the ellipse, respectively.
#
# Lithotype decision functions operate in the latent Gaussian domain [-3,3]×[-3,3].
# Since each GRF is N(0,1), approximately 99.7% of its values lie within ±3σ,
# making ±3 a natural “full” range for defining splits or shapes.

config = {
    "root": {
        "type": "decision",
        "func": ellipse,
        "args": {
            "key1": "Z1",
            "key2": "Z2",
            "c1": 0,
            "c2": 0,
            "s1": 2.5,
            "s2": 0.8,
            "angle": -45,
        },
        "yes_branch": "phase1",
        "no_branch": "phase0",
    },
    "phase1": {"type": "leaf", "action": 1},
    "phase0": {"type": "leaf", "action": 0},
}

###############################################################################
# With the tree configuration ready, we can create the PGS object as normal,
# passing our domain size and spatial random fields. When generating the
# plurigaussian field, we pass the tree configuration so GSTools knows which
# PGS process to follow.

pgs = gs.PGS(dim, [field1, field2])

P = pgs(tree=config)

###############################################################################
# We can also compute the equivalent spatial lithotype.
#
# NB: If we want to compute L before P, we must pass the tree config to
# `pgs.compute_lithotype(tree=config)`.

L = pgs.compute_lithotype()

###############################################################################
# Finally, we plot `P` as well as the equivalent spatial lithotype `L`.

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

im0 = axs[0].imshow(L, cmap="copper", origin="lower")
axs[0].set_title("L")
im1 = axs[1].imshow(P, cmap="copper", origin="lower")
axs[1].set_title("P")

###############################################################################
#
# .. image:: ../../pics/2d_tree_pgs.png
#    :width: 400px
#    :align: center
