r"""
Reproducing Mariethoz et al. (2010) Figure 8: non-stationary co-simulation
-----------------------------------------------------------------------------

Figure 8 co-simulates a channel facies together with an auxiliary orientation
variable. Both the training image and the simulation grid carry the auxiliary
variable, so the joint scan is forced to match the local channel direction —
the auxiliary variable acts as the non-stationarity carrier instead of an
explicit rotation map.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

# 1. Create the Synthetic TI (250x250)
ti_size = 250
gx_ti, gy_ti = np.meshgrid(
    np.arange(ti_size), np.arange(ti_size), indexing="ij"
)

# Primary variable: channels rotating as a function of X
# theta varies from 0 (horizontal) to pi/2 (vertical) along X
theta_ti = (gx_ti / float(ti_size - 1)) * (np.pi / 2.0)
# create a simple pattern that rotates
channels_ti = (
    (np.sin(gx_ti * np.cos(theta_ti) * 0.5 + gy_ti * np.sin(theta_ti) * 0.5))
    > 0
).astype(float)

# Secondary variable: X coordinate normalized
context_ti = gx_ti / float(ti_size - 1)

# Assemble multivariate TI using Variable list.
# From paper: n1 = 30 for primary, n2 = 1 for secondary.
ti = gs.TrainingImage(
    [
        gs.Variable(
            "channels",
            channels_ti,
            categorical=True,
            weight=0.5,
            n_neighbors=30,
        ),
        gs.Variable(
            "context",
            context_ti,
            categorical=False,
            weight=0.5,
            distance="l1",
            n_neighbors=1,
        ),
    ]
)

# 2. Setup the MPS Model
# We use scan_fraction=0.5 (half the TI) and a tight threshold=0.01
# since we have a perfect synthetic correlation to match.
model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

# 3. Setup the Simulation Grid (Exhaustive Secondary Variable)
# The simulation grid is 250x250.
sg_size = 250
xs_sg = np.arange(sg_size, dtype=float)
ys_sg = np.arange(sg_size, dtype=float)
gx_sg, gy_sg = np.meshgrid(xs_sg, ys_sg, indexing="ij")

# Secondary variable in SG: normalized Y coordinate
# We want horizontal at context=0 (which is X=0 in TI), vertical at context=1 (X=250 in TI)
# The paper says: zeros are at the bottom, ones are on top.
context_sg = gy_sg / float(sg_size - 1)

# To use this as an exhaustive secondary variable, we "condition" the simulation
# with the context variable at every single point.
cond_pos = [gx_sg.flatten(), gy_sg.flatten()]
cond_val = {
    # np.nan means "unconditioned" for the channels variable
    "channels": np.full(sg_size * sg_size, np.nan),
    "context": context_sg.flatten(),
}

ds = gs.DirectSampling(model)
ds.set_condition(cond_pos, cond_val)

# 4. Run the simulation
print(f"Simulating multivariate non-stationary field ({sg_size}x{sg_size})...")
# Note: we use num_threads=1 because we found the inner loop thread overhead is high!
fields = ds([xs_sg, ys_sg], seed=42, num_threads=1)
channels_sim = fields["channels"]
context_sim = fields["context"]

# 5. Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(channels_ti, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Training image, variable 1")

axes[0, 1].imshow(context_ti, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[0, 1].set_title("b) Training image, variable 2")

axes[1, 0].imshow(channels_sim, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulation, variable 1")

axes[1, 1].imshow(context_sim, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[1, 1].set_title("d) Simulation, variable 2")

fig.tight_layout()
plt.savefig("mariethoz_fig8_reproduction.png")
print("Saved mariethoz_fig8_reproduction.png")
