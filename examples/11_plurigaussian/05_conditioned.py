"""
Creating conditioned PGS
------------------------

In case we have knowledge about values of the PGS in certain areas, we should
incorporate that knowledge. We can do this by using conditional fields, which
are created by combining kriged fields with SRFs. This can be done quite easily
with GSTools. For more details, have a look at the kriging and conditional
field examples.

In this example, we assume that we know the categorical values of the PGS field
to be 2 in the lower left 20 by 20 grid cells.

Warning: Using PGS for conditioning fields is still a beta feature.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2
# no. of cells in both dimensions
N = [100, 80]

# grid
x = np.arange(N[0])
y = np.arange(N[1])

###############################################################################
# Now we want to read the known data in order to condition the PGS on them.
# Normally, we would probably get the data in a different format, but in order
# to keep the dependencies for this example to a minimum, we will simpy use a
# numpy format. After reading in the data, we will have a quick look at how the
# data looks like.
# We'll see the first few values, which are all 1. This value is not very
# important. However, it should be in the range of SRF values, to not mess
# with the kriging. The known value of the PGS, namely 2, will be set for the
# lithotypes field, which will map it the the conditioned SRF values of 1 at given
# positions.

cond_data = np.load("conditional_values.npz")
cond_pos = cond_data["cond_pos"]
cond_val = cond_data["cond_val"]
print(f"first 5 conditional positions:\n{cond_pos[:, :5]}")
print(f"first 5 conditional values:\n{cond_val[:5]}")

###############################################################################
# With the conditional values ready, we can now set up the covariance model
# for the kriging. This knowledge has to normally be inferred, but here we just
# assume that we know the convariance structure of the underlying field.
# For better visualization, we use `Simple` kriging with a mean value of 0.

model = gs.Gaussian(dim=dim, var=1, len_scale=[10, 5], angles=np.pi / 8)
krige = gs.krige.Simple(model, cond_pos=cond_pos, cond_val=cond_val, mean=0)
cond_srf = gs.CondSRF(krige)
cond_srf.set_pos([x, y], "structured")

###############################################################################
# Now that the conditioned field class is set up, we can generate SRFs
# conditioned on our previous knowledge. We'll do that for the two SRFs needed
# for the PGS, and then we will also set up the PGS generator. Next, we'll
# use a little helper method, which can transform the coordinates from the SRFs
# to the lithotypes field. This helps us set up the area around the conditioned value
# `cond_val`.

field1 = cond_srf(seed=484739)
field2 = cond_srf(seed=45755894)

pgs = gs.PGS(dim, [field1, field2])

M = [100, 80]

# size of the rectangle
rect = [40, 32]

lithotypes = np.zeros(M)
# calculate grid axes of the lithotypes field
pos_lith = pgs.calc_lithotype_axes(lithotypes.shape)
# transform conditioned SRF value to lithotypes index
pos_lith_ind = pgs.transform_coords(
    lithotypes.shape, [cond_val[0], cond_val[0]]
)

# conditioned category of 2 around the conditioned values' positions
lithotypes[
    pos_lith_ind[0] - 5 : pos_lith_ind[0] + 5,
    pos_lith_ind[1] - 5 : pos_lith_ind[1] + 5,
] = 2

###############################################################################
# With the two SRFs and the lithotypes ready, we can create the actual PGS.

P = pgs(lithotypes)

###############################################################################
# Finally, we can plot the PGS, but we will also show the lithotypes and the
# two original Gaussian SRFs. We will set the colours of the SRF correlation
# scatter plot to be the sum of their respective position tuples (x+y), to get
# a feeling for which point corresponds to which position. The more blue the
# points, the smaller the sum is. We can nicely see that many blue points
# gather in the highlighted rectangle of the lithotypes where the categorical
# value of 2 is set.

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(field1, cmap="copper", vmin=-3, vmax=2, origin="lower")
axs[0, 1].imshow(field2, cmap="copper", vmin=-3, vmax=2, origin="lower")
axs[1, 0].scatter(
    field1.flatten(),
    field2.flatten(),
    s=0.1,
    c=(x.reshape((len(x), 1)) + y.reshape((1, len(y))).flatten()),
)
axs[1, 0].pcolormesh(
    pos_lith[0], pos_lith[1], lithotypes.T, alpha=0.3, cmap="copper"
)
axs[1, 1].imshow(P, cmap="copper", origin="lower")

plt.tight_layout()
plt.show()

###############################################################################
# With all this set up, we can easily create an ensemble of PGS, which conform
# to the conditional values

seed = gs.random.MasterRNG(20170519)

ens_no = 9
fields1 = []
fields2 = []
Ps = []
for i in range(ens_no):
    fields1.append(cond_srf(seed=seed()))
    fields2.append(cond_srf(seed=seed()))
    pgs = gs.PGS(dim, [fields1[-1], fields2[-1]])
    Ps.append(pgs(lithotypes))

fig, axs = plt.subplots(3, 3)
cnt = 0
for i in range(int(np.sqrt(ens_no))):
    for j in range(int(np.sqrt(ens_no))):
        axs[i, j].imshow(Ps[cnt], cmap="copper", origin="lower")

        cnt += 1

plt.tight_layout()
plt.show()
