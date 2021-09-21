"""
Creating an Ensemble of Fields
------------------------------

Creating an ensemble of random fields would also be
a great idea. Let's reuse most of the previous code.

We will set the position tuple `pos` before generation to reuse it afterwards.
"""

import matplotlib.pyplot as pt
import numpy as np

import gstools as gs

x = y = np.arange(100)

model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model)
srf.set_pos([x, y], "structured")

###############################################################################
# This time, we did not provide a seed to :any:`SRF`, as the seeds will used
# during the actual computation of the fields. We will create four ensemble
# members, for better visualisation, save them in to srf class and in a first
# step, we will be using the loop counter as the seeds.

ens_no = 4
for i in range(ens_no):
    srf(seed=i, store=f"field{i}")

###############################################################################
# Now let's have a look at the results. We can access the fields by name or
# index:

fig, ax = pt.subplots(2, 2, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(ens_no):
    ax[i].imshow(srf[i].T, origin="lower")
pt.show()

###############################################################################
# Using better Seeds
# ^^^^^^^^^^^^^^^^^^
#
# It is not always a good idea to use incrementing seeds. Therefore GSTools
# provides a seed generator :any:`MasterRNG`. The loop, in which the fields are
# generated would then look like

from gstools.random import MasterRNG

seed = MasterRNG(20170519)
for i in range(ens_no):
    srf(seed=seed(), store=f"better_field{i}")
