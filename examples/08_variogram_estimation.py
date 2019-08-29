import os
from shutil import rmtree
import zipfile
import urllib.request
import numpy as np
import matplotlib.pyplot as pt
from gstools import (
    vario_estimate_unstructured,
    vario_estimate_structured,
    Exponential,
)
from gstools.covmodel.plot import plot_variogram


def download_herten():
    # download the data, warning: its about 250MB
    print("Downloading Herten data")
    data_filename = "data.zip"
    data_url = "http://store.pangaea.de/Publications/Bayer_et_al_2015/Herten-analog.zip"
    urllib.request.urlretrieve(data_url, "data.zip")

    with zipfile.ZipFile(data_filename, "r") as zf:
        zf.extract(
            os.path.join("Herten-analog", "sim-big_1000x1000x140", "sim.vtk")
        )


def download_scripts():
    # download a script for file conversion
    print("Downloading scripts")
    tools_filename = "scripts.zip"
    tool_url = (
        "http://store.pangaea.de/Publications/Bayer_et_al_2015/tools.zip"
    )
    urllib.request.urlretrieve(tool_url, tools_filename)

    with zipfile.ZipFile(tools_filename, "r") as zf:
        zf.extract(os.path.join("tools", "vtk2gslib.py"))


def create_unstructured_grid(x_s, y_s):
    x_u, y_u = np.meshgrid(x_s, y_s)
    len_unstruct = len(x_s) * len(y_s)
    x_u = np.reshape(x_u, len_unstruct)
    y_u = np.reshape(y_u, len_unstruct)
    return x_u, y_u


###############################################################################
# data preparation ############################################################
###############################################################################

# uncomment these two function calls, in case the data was already downloaded
# and you want to execute this script multiple times. But don't forget to
# comment out the cleanup code at the end of this script.
download_herten()
download_scripts()

# import the downloaded conversion script
from tools.vtk2gslib import vtk2numpy

# load the Herten aquifer with the downloaded vtk2numpy routine
print("Loading data")
herten, grid = vtk2numpy(
    os.path.join("Herten-analog", "sim-big_1000x1000x140", "sim.vtk")
)

# conductivity values per fazies from the supplementary data
cond = np.array(
    [
        2.50e-04,
        2.30e-04,
        6.10e-05,
        2.60e-02,
        1.30e-01,
        9.50e-02,
        4.30e-05,
        6.00e-07,
        2.30e-03,
        1.40e-04,
    ]
)

# asign the conductivities to the facies
herten_cond = cond[herten]

# integrate over the vertical axis, calculate transmissivity
herten_log_trans = np.log(np.sum(herten_cond, axis=2) * grid["dz"])

# create a structured grid on which the data is defined
x_s = np.arange(grid["ox"], grid["nx"] * grid["dx"], grid["dx"])
y_s = np.arange(grid["oy"], grid["ny"] * grid["dy"], grid["dy"])

pt.imshow(herten_log_trans.T, origin="lower", aspect="equal")
pt.show()

# create an unstructured grid for the variogram estimation
x_u, y_u = create_unstructured_grid(x_s, y_s)

###############################################################################
# estimate the variogram on an unstructured grid ##############################
###############################################################################

bins = np.linspace(0, 10, 50)
print("Estimating unstructured variogram")
bin_center, gamma = vario_estimate_unstructured(
    (x_u, y_u),
    herten_log_trans.reshape(-1),
    bins,
    sampling_size=2000,
    sampling_seed=19920516,
)

# fit an exponential model
fit_model = Exponential(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)

pt.plot(bin_center, gamma)
plot_variogram(fit_model, x_max=bins[-1])

###############################################################################
# estimate the variogram on a structured grid #################################
###############################################################################

# estimate the variogram on a structured grid
# use only every 10th value, otherwise calculations would take very long
x_s_skip = x_s[::10]
y_s_skip = y_s[::10]
herten_trans_skip = herten_log_trans[::10, ::10]

print("Estimating structured variograms")
gamma_x = vario_estimate_structured(herten_trans_skip, direction="x")
gamma_y = vario_estimate_structured(herten_trans_skip, direction="y")

x_plot = x_s_skip[:21]
y_plot = y_s_skip[:21]
# fit an exponential model
fit_model_x = Exponential(dim=2)
fit_model_x.fit_variogram(x_plot, gamma_x[:21], nugget=False)
fit_model_y = Exponential(dim=2)
fit_model_y.fit_variogram(y_plot, gamma_y[:21], nugget=False)

line, = pt.plot(bin_center, gamma, label="estimated variogram (isotropic)")
pt.plot(
    bin_center,
    fit_model.variogram(bin_center),
    color=line.get_color(),
    linestyle="--",
    label="exp. variogram (isotropic)",
)

line, = pt.plot(x_plot, gamma_x[:21], label="estimated variogram in x-dir")
pt.plot(
    x_plot,
    fit_model_x.variogram(x_plot),
    color=line.get_color(),
    linestyle="--",
    label="exp. variogram in x-dir",
)

line, = pt.plot(y_plot, gamma_y[:21], label="estimated variogram in y-dir")
pt.plot(
    y_plot,
    fit_model_y.variogram(y_plot),
    color=line.get_color(),
    linestyle="--",
    label="exp. variogram in y-dir",
)

pt.legend()
pt.show()

print("semivariogram model (isotropic):\n", fit_model)
print("semivariogram model (in x-dir.):\n", fit_model_x)
print("semivariogram model (in y-dir.):\n", fit_model_y)

###############################################################################
# creating a SRF from the Herten parameters ###################################
###############################################################################

from gstools import SRF

srf = SRF(fit_model, seed=19770928)
print("Calculating SRF")
new_herten = srf((x_s, y_s), mesh_type="structured")

pt.imshow(new_herten.T, origin="lower")
pt.show()

###############################################################################
# cleanup #####################################################################
###############################################################################

# comment all in case you want to keep the data for playing around with it
os.remove("data.zip")
os.remove("scripts.zip")
rmtree("Herten-analog")
rmtree("tools")
