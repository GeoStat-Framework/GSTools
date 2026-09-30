r"""
Reproducing Mariethoz et al. (2010) Figure 4: bivariate joint simulation
---------------------------------------------------------------------------

Figure 4 simulates two dependent variables at once: the Strebelle channel
facies and a smoothed, noise-perturbed derivative of it. Direct Sampling
copies both values from the *same* training image cell at every node, so the
cross-variable relationship is reproduced exactly rather than being imposed
by a coregionalization model.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

import gstools as gs

# 1. Prepare the Training Image
# Load the Strebelle training image (Variable 1)
TI_URL = (
    "https://raw.githubusercontent.com/GeostatsGuy/"
    "GeoDataSets/master/MPS_Training_image_and_Realizations_500.npz"
)
CACHE = "mps_strebelle.npz"
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(TI_URL, CACHE)

ti_full = np.load(CACHE)["array1"].astype(float)
# The figure shows a 250x250 subset
ti_var1 = ti_full[:250, :250]

# Generate Variable 2 (e.g. resistivity)
# Paper: "smoothing variable 1 using a moving average with a window made of
# the 500 closest nodes and then adding an uncorrelated white noise [0, 0.5]"
# A 23x23 square window contains 529 nodes, which is a good approximation.
smoothed = uniform_filter(ti_var1, size=23, mode="reflect")

np.random.seed(42)
noise = np.random.uniform(0, 0.5, size=ti_var1.shape)
ti_var2 = smoothed + noise

# 2. Assemble Multivariate TI using Variable list.
# Paper says Variable 2 uses distance (4), which is the weighted RMSE ("l2" in GSTools).
# n1 = 30, n2 = 30, t = 0.01, w1 = 0.5, w2 = 0.5
ti = gs.TrainingImage(
    [
        gs.Variable(
            "facies",
            ti_var1,
            categorical=True,
            weight=0.5,
            distance="l1",
            n_neighbors=30,
        ),
        gs.Variable(
            "resistivity",
            ti_var2,
            categorical=False,
            weight=0.5,
            distance="l2",
            n_neighbors=30,
        ),
    ]
)

# 3. Setup the MPS Model
model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

# 4. Run unconditional simulation
ds = gs.DirectSampling(model)

grid_size = 250
x = y = np.arange(grid_size, dtype=float)

print(f"Simulating unconditional bivariate field ({grid_size}x{grid_size})...")
fields = ds([x, y], seed=123, num_threads=1)

sim_var1 = fields["facies"]
sim_var2 = fields["resistivity"]

# 5. Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# a) TI var 1
axes[0, 0].imshow(ti_var1, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Training image, variable 1")

# b) TI var 2
im_b = axes[0, 1].imshow(
    ti_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5
)
axes[0, 1].set_title("b) Training image, variable 2")
plt.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04)

# c) Sim var 1
axes[1, 0].imshow(sim_var1, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulation, variable 1")

# d) Sim var 2
im_d = axes[1, 1].imshow(
    sim_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5
)
axes[1, 1].set_title("d) Simulation, variable 2")
plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.tight_layout()
plt.savefig("mariethoz_fig4_reproduction.png")
print("Saved mariethoz_fig4_reproduction.png")
