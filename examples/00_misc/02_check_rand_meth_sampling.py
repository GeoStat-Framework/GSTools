"""
Check Random Sampling
---------------------
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gstools as gs


def norm_rad(vec):
    """Direction on the unit sphere."""
    vec = np.array(vec, ndmin=2)
    norm = np.zeros(vec.shape[1])
    for i in range(vec.shape[0]):
        norm += vec[i] ** 2
    norm = np.sqrt(norm)
    return np.einsum("j,ij->ij", 1 / norm, vec), norm


def plot_rand_meth_samples(generator):
    """Plot the samples of the rand meth class."""
    norm, rad = norm_rad(generator._cov_sample)

    fig = plt.figure(figsize=(10, 4))

    if generator.model.dim == 3:
        ax = fig.add_subplot(121, projection=Axes3D.name)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color="b", alpha=0.1)
        ax.scatter(norm[0], norm[1], norm[2])
    elif generator.model.dim == 2:
        ax = fig.add_subplot(121)
        u = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(u)
        y = np.sin(u)
        ax.plot(x, y, color="b", alpha=0.1)
        ax.scatter(norm[0], norm[1])
        ax.set_aspect("equal")
    else:
        ax = fig.add_subplot(121)
        ax.bar(-1, np.sum(np.isclose(norm, -1)), color="C0")
        ax.bar(1, np.sum(np.isclose(norm, 1)), color="C0")
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(("-1", "1"))
    ax.set_title("Direction sampling")

    ax = fig.add_subplot(122)
    x = np.linspace(0, 10 / generator.model.integral_scale)
    y = generator.model.spectral_rad_pdf(x)
    ax.plot(x, y, label="radial spectral density")
    sample_in = np.sum(rad <= np.max(x))
    ax.hist(rad[rad <= np.max(x)], bins=sample_in // 50, density=True)
    ax.set_xlim([0, np.max(x)])
    ax.set_title("Radius samples shown {}/{}".format(sample_in, len(rad)))
    ax.legend()
    plt.show()


model = gs.Stable(dim=3, alpha=1.5)
srf = gs.SRF(model, seed=2020)
plot_rand_meth_samples(srf.generator)
