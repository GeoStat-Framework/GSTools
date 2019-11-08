# Welcome to GSTools

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313628.svg)](https://doi.org/10.5281/zenodo.1313628)
[![PyPI version](https://badge.fury.io/py/gstools.svg)](https://badge.fury.io/py/gstools)
[![Build Status](https://travis-ci.org/GeoStat-Framework/GSTools.svg?branch=master)](https://travis-ci.org/GeoStat-Framework/GSTools)
[![Build status](https://ci.appveyor.com/api/projects/status/oik6h65n0xdy4h4j/branch/master?svg=true)](https://ci.appveyor.com/project/GeoStat-Framework/gstools/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/GSTools/badge.svg?branch=master)](https://coveralls.io/github/GeoStat-Framework/GSTools?branch=master)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://geostat-framework.readthedocs.io/projects/gstools/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gstools.png" alt="GSTools-LOGO" width="251px"/>
</p>


## Purpose

<img align="right" width="450" src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/demonstrator.png" alt="">

GeoStatTools provides geostatistical tools for various purposes:
- random field generation
- conditioned field generation
- incompressible random vector field generation
- simple and ordinary kriging
- variogram estimation and fitting
- many readily provided and even user-defined covariance models
- plotting and exporting routines


## Installation

The package can be installed via [pip][pip_link] on Windows, Linux and Mac.
On Windows you can install [WinPython][winpy_link] to get
Python and pip running. Also [conda provides pip support][conda_pip].
Install GSTools by typing the following into the command prompt:

    pip install gstools

To get the latest development version you can install it directly from GitHub:

    pip install https://github.com/GeoStat-Framework/GSTools/archive/develop.zip

To enable the OpenMP support, you have to provide a C compiler, Cython and OpenMP.
To get all other dependencies, it is recommended to first install gstools once
in the standard way just decribed.
Then use the following command:

    pip install --global-option="--openmp" gstools

Or for the development version:

    pip install --global-option="--openmp" https://github.com/GeoStat-Framework/GSTools/archive/develop.zip

If something went wrong during installation, try the [``-I`` flag from pip][pipiflag].


## Citation

At the moment you can cite the Zenodo code publication of GSTools:

> Sebastian Müller, & Lennart Schüler. (2019, October 1). GeoStat-Framework/GSTools: Reverberating Red (Version v1.1.0). Zenodo. http://doi.org/10.5281/zenodo.3468230

A publication for the GeoStat-Framework is in preperation.


## Documentation for GSTools

You can find the documentation under [geostat-framework.readthedocs.io][doc_link].


### Tutorials and Examples

The documentation also includes some [tutorials][tut_link], showing the most important use cases of GSTools, which are

- [Random Field Generation][tut1_link]
- [The Covariance Model][tut2_link]
- [Variogram Estimation][tut3_link]
- [Random Vector Field Generation][tut4_link]
- [Kriging][tut5_link]
- [Conditioned random field generation][tut6_link]
- [Field transformations][tut7_link]

Some more examples are provided in the examples folder.


## Spatial Random Field Generation

The core of this library is the generation of spatial random fields. These fields are generated using the randomisation method, described by [Heße et al. 2014][rand_link].

[rand_link]: https://doi.org/10.1016/j.envsoft.2014.01.013


### Examples

#### Gaussian Covariance Model

This is an example of how to generate a 2 dimensional spatial random field with a gaussian covariance model.

```python
from gstools import SRF, Gaussian
import matplotlib.pyplot as plt
# structured field with a size 100x100 and a grid-size of 1x1
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
srf((x, y), mesh_type='structured')
srf.plot()
```
<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gau_field.png" alt="Random field" width="600px"/>
</p>

A similar example but for a three dimensional field is exported to a [VTK](https://vtk.org/) file, which can be visualized with [ParaView](https://www.paraview.org/) or [PyVista](https://docs.pyvista.org) in Python:

```python
from gstools import SRF, Gaussian
import matplotlib.pyplot as pt
# structured field with a size 100x100x100 and a grid-size of 1x1x1
x = y = z = range(100)
model = Gaussian(dim=3, var=0.6, len_scale=20)
srf = SRF(model)
srf((x, y, z), mesh_type='structured')
srf.vtk_export('3d_field') # Save to a VTK file for ParaView

mesh = srf.to_pyvista() # Create a PyVista mesh for plotting in Python
mesh.threshold_percent(0.5).plot()
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/3d_gau_field.png" alt="3d Random field" width="600px"/>
</p>


## Estimating and Fitting Variograms

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.

All covariance models can be used to fit given variogram data by a simple interface.

### Example

This is an example of how to estimate the variogram of a 2 dimensional unstructured field and estimate the parameters of the covariance
model again.

```python
import numpy as np
from gstools import SRF, Exponential, Stable, vario_estimate_unstructured
# generate a synthetic field with an exponential model
x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = Exponential(dim=2, var=2, len_scale=8)
srf = SRF(model, mean=0, seed=19970221)
field = srf((x, y))
# estimate the variogram of the field with 40 bins
bins = np.arange(40)
bin_center, gamma = vario_estimate_unstructured((x, y), field, bins)
# fit the variogram with a stable model. (no nugget fitted)
fit_model = Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)
# output
ax = fit_model.plot(x_max=40)
ax.plot(bin_center, gamma)
print(fit_model)
```

Which gives:

```python
Stable(dim=2, var=1.92, len_scale=8.15, nugget=0.0, anis=[1.], angles=[0.], alpha=1.05)
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/exp_vario_fit.png" alt="Variogram" width="600px"/>
</p>


## Kriging and Conditioned Random Fields

An important part of geostatistics is Kriging and conditioning spatial random
fields to measurements. With conditioned random fields, an ensemble of field realizations with their variability depending on the proximity of the measurements can be generated.

### Example
For better visualization, we will condition a 1d field to a few "measurements", generate 100 realizations and plot them:

```python
import numpy as np
from gstools import Gaussian, SRF
import matplotlib.pyplot as plt

# conditions
cond_pos = [0.3, 1.9, 1.1, 3.3, 4.7]
cond_val = [0.47, 0.56, 0.74, 1.47, 1.74]

gridx = np.linspace(0.0, 15.0, 151)

# spatial random field class
model = Gaussian(dim=1, var=0.5, len_scale=2)
srf = SRF(model)
srf.set_condition(cond_pos, cond_val, "ordinary")

# generate the ensemble of field realizations
fields = []
for i in range(100):
    fields.append(srf(gridx, seed=i))
    plt.plot(gridx, fields[i], color="k", alpha=0.1)
plt.scatter(cond_pos, cond_val, color="k")
plt.show()
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/cond_ens.png" alt="Conditioned" width="600px"/>
</p>

## User Defined Covariance Models

One of the core-features of GSTools is the powerful
[CovModel][cov_link]
class, which allows to easy define covariance models by the user.

### Example

Here we re-implement the Gaussian covariance model by defining just a
[correlation][cor_link] function, which takes a non-dimensional distance ``h = r/l``:

```python
from gstools import CovModel
import numpy as np
# use CovModel as the base-class
class Gau(CovModel):
    def cor(self, h):
        return np.exp(-h**2)
```

And that's it! With ``Gau`` you now have a fully working covariance model,
which you could use for field generation or variogram fitting as shown above.

Have a look at the [documentation ][doc_link] for further information on incorporating
optional parameters and optimizations.


## Incompressible Vector Field Generation

Using the original [Kraichnan method][kraichnan_link], incompressible random
spatial vector fields can be generated.


### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
x = np.arange(100)
y = np.arange(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model, generator='VectorField')
srf((x, y), mesh_type='structured', seed=19841203)
srf.plot()
```

yielding

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/vec_srf_tut_gau.png" alt="vector field" width="600px"/>
</p>


[kraichnan_link]: https://doi.org/10.1063/1.1692799


## VTK/PyVista Export

After you have created a field, you may want to save it to file, so we provide
a handy [VTK][vtk_link] export routine using the `.vtk_export()` or you could
create a VTK/PyVista dataset for use in Python with to `.to_pyvista()` method:

```python
from gstools import SRF, Gaussian
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
srf((x, y), mesh_type='structured')
srf.vtk_export("field") # Saves to a VTK file
mesh = srf.to_pyvista() # Create a VTK/PyVista dataset in memory
mesh.plot()
```

Which gives a RectilinearGrid VTK file ``field.vtr`` or creates a PyVista mesh
in memory for immediate 3D plotting in Python.


## Requirements:

- [NumPy >= 1.14.5](https://www.numpy.org)
- [SciPy >= 1.1.0](https://www.scipy.org/scipylib)
- [hankel >= 0.3.6](https://github.com/steven-murray/hankel)
- [emcee >= 3.0.0](https://github.com/dfm/emcee)
- [pyevtk](https://bitbucket.org/pauloh/pyevtk)
- [six](https://github.com/benjaminp/six)

### Optional

- [matplotlib](https://matplotlib.org)
- [pyvista](https://docs.pyvista.org/)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[LGPLv3][license_link] © 2018-2019

[pip_link]: https://pypi.org/project/gstools
[conda_pip]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages
[pipiflag]: https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i
[winpy_link]: https://winpython.github.io/
[license_link]: https://github.com/GeoStat-Framework/GSTools/blob/master/LICENSE
[cov_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/covmodel.base.html#gstools.covmodel.base.CovModel
[stable_link]: https://en.wikipedia.org/wiki/Stable_distribution
[doc_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/
[tut_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorials.html
[tut1_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_01_srf.html
[tut2_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_02_cov.html
[tut3_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_03_vario.html
[tut4_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_04_vec_field.html
[tut5_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_05_kriging.html
[tut6_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_06_conditioning.html
[tut7_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_07_transformations.html
[cor_link]: https://en.wikipedia.org/wiki/Autocovariance#Normalization
[vtk_link]: https://www.vtk.org/
