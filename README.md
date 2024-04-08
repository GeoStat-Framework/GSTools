# Welcome to GSTools
[![GMD](https://img.shields.io/badge/GMD-10.5194%2Fgmd--15--3161--2022-orange)](https://doi.org/10.5194/gmd-15-3161-2022)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313628.svg)](https://doi.org/10.5281/zenodo.1313628)
[![PyPI version](https://badge.fury.io/py/gstools.svg)](https://badge.fury.io/py/gstools)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gstools.svg)](https://anaconda.org/conda-forge/gstools)
[![Build Status](https://github.com/GeoStat-Framework/GSTools/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/GeoStat-Framework/GSTools/actions)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/GSTools/badge.svg?branch=main)](https://coveralls.io/github/GeoStat-Framework/GSTools?branch=main)
[![Documentation Status](https://readthedocs.org/projects/gstools/badge/?version=latest)](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/gstools.png" alt="GSTools-LOGO" width="251px"/>
</p>

<p align="center"><b>Get in Touch!</b></p>
<p align="center">
<a href="https://github.com/GeoStat-Framework/GSTools/discussions"><img src="https://img.shields.io/badge/GitHub-Discussions-f6f8fa?logo=github&style=flat" alt="GH-Discussions"/></a>
<a href="https://swung.slack.com/messages/gstools"><img src="https://img.shields.io/badge/Swung-Slack-4A154B?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAABmJLR0QA%2FwD%2FAP%2BgvaeTAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAAB3RJTUUH5AYaFSENGSa5qgAABmZJREFUSMeFlltsVNcVhr%2B1z5m7Zzy%2BxaBwcQrGQOpgCAkKtSBQIqJepKhPBULpQ6sKBVWVKqXtSy%2BR0qYXqa2qRmlDCzjBEZGKUCK1TWqlNiGIEKDQBtf4Fki4OIxnxrex53LOXn2YwbjEtOvlHG3tvX%2Btf%2B21%2Fl%2BYJ1QVEbn1vwLYBWwCVgG1lW0ZoA%2FoAQ6LSP%2BdZ%2BeGzAMiIqK%2Bem0GpxNYVeBj3j2b4NCfM2QnfAAaa11al4fZuCZK24owQJ9v%2BbLryIVbd9wVSNUaEWNVtQPYfXHmAD0T32ZJeBM1Q8d0zzMDUpMwAFgLJU%2BxClURw9NfqedLWxMAHSKyR1WNiNhPAM0B6c%2FbdPORTLuOeUMSNkmMBHgyeo32bwwRDMh8bDM%2BZVl0j6uvPrdYknFnSESWzwUzt%2BkyVlUHx7zh5j%2BmPkXBjosjLkWdominiMQ%2BoiEZxuq8OFRXGXJ5K5%2Fde5nha8VlqjooIlZVBcBUiqeqemjGppd1ptfhSpS8pmmN7GVf4whPNY4Di9m%2BMcR03nK3sBbCQeFbv7gBsExVOyp3l6nz1VtjcM4fTK3Uok5IXtPsrHuPevcBXk8d4dWPX6I%2BsIB9wf1s%2B2Y%2FVbFynUIBIeDeplIECiXl5Iv3kbLogogRgbWukfNumT%2FnlYszBxj3hwXg0cQvqXcfYNu5tVyYPE%2B1G8dXn%2BfW72fH49U8sSlOPGr4SccoF4cKs3WzFrY%2BFCMUNmz%2Ba0aeWR1l15JwJ7DaVPpk1YnJ7xIxtQRNjDXRvTx%2F9ef0Tl0g6SYQhAlvmkH%2Fgv74qUaiTSG8ewJ0%2FGgRK5aG8Cts5ouWDa1RxoDRovK9i9MAq1S12QA7b5ROUdBxBIeQ1ACG49m%2FEXPis7Qk3ChHbx6Qw1dgXVeWB7uyDOctP%2Fx6w2zdrIVIyFCyiq8wXlJOZzyAXQbY%2FGGhC8EAilJ%2BVg7ufxU6IAHeSvewfQEadiDuCr%2B6NE1LU4hwUFAF1xFGRkvEjVDlgiPwVqoEsNkAq0ZKp3EIYrFM2xGm7Uc8u%2FzXjHkTmHIHoCiDM73E3IIsDCtRV3gn7QHQ0hTCt0ooKLw%2FWCAM1AcNISOcHSsBrDRAbc7eQMQBFFciHM18kaZIMz3r%2F0HO5mazytsiw%2FmTtCYiGGCkQlltwkEVjMDVmyUA6oIGR%2BDGjAWoM3f2giHAhH%2BFI5nPsDrWxqWNE9S4tUz5k1S7cQ5df4k9S6qY9JRipXtr4w5WQYH0eHkWrqxy8FTn3AvpmFmIqj%2B76EiQjNfHH1JNWFKc3vABj9V9npw%2FRXfmBNsaoTRnRAQDAgqqMJr1KBWUtUmHaR8WRgzAqAH6FgYexqd4R2Yuns5wcLSFK4U36bj%2FdbbUbGdoZoCi3uS%2Bqtt73TlNWygpqXGfZTGXnKesrwkA9BmgZ0noMZT5R0tQ4hzLfo4rhS46W%2F%2BCAn3T7%2BhDySiWMl2RkHArP8dAesKjPixYVbbUBwB6DHB4QWADIamuHPtkhE0t3ZP7ANhe9zgvXP2dfK0pymRJmQLiEYNW6mEVljYGuDzlkwwaHq51AQ4bERkAetvjP2XCT6H480AJeZsB4N7QYt7OnuSROtRXJV2wNNS4qIJvlbUtERJxhxcv5%2FlNWwygV0QGyzKBv%2FP%2ByFfZXf%2ButoR3UuXcS95mKNgxSjpN3qZZFHwUgFPjx5n2c9wo9ktrtcOZtMeWB2NEw4b2thivPLuIS1M%2BAzmrTy4O4ys7Zv1B5fsnVdWCr7PxYf7vej73ex2YeU1VVY9nu7ShG63vRo%2Fe%2FK1%2B518FbXkjo3OjO1XU2LFRzRZ9VdWDczFQ1VsCOHgpd1G%2FcG6jHrj2vPbn%2BjVdHNfr%2BRH92eXva2MPuvxEQpe%2BHdEnzm%2FQf4%2BrRo%2BldMUbGd393oS2dWU0cDSlw1OequrALVG9Q8rLsquqg2OlzLL2Myu1N5eShgB4CjEnSMSJYrX8Oj0t8UH7NMnX0iSDwmhBWRl3tKs9IcmgGRSRZqtqzFwpL4uWWKvWiMjyZKC24%2F1HbsrLn95Pwk3gCpS0yIw%2Fg6clPC2RLc3QmzvJupoARQsvrItxZmtSkkFz6E6Q%2F2m3PFta44jbCaw%2BO3GK7uybnJs8xfXC1fLYCdTz9NIfsCS0mYVhAHp9ZYdr5J%2F%2F127dxUA2AzuBzRUDWVfZlq4YyG6gs9ImdzWQ%2FwFNRlgCFdG5bAAAAABJRU5ErkJggg%3D%3D" alt="Slack-Swung"/></a>
<a href="https://gitter.im/GeoStat-Framework/GSTools"><img src="https://img.shields.io/badge/Gitter-GeoStat--Framework-ed1965?logo=gitter&style=flat" alt="Gitter-GSTools"/></a>
<a href="mailto:info@geostat-framework.org"><img src="https://img.shields.io/badge/Email-GeoStat--Framework-468a88?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbDpzcGFjZT0icHJlc2VydmUiIHdpZHRoPSI1MDAiIGhlaWdodD0iNTAwIj48cGF0aCBkPSJNNDQ4IDg4SDUyYy0yNyAwLTQ5IDIyLTQ5IDQ5djIyNmMwIDI3IDIyIDQ5IDQ5IDQ5aDM5NmMyNyAwIDQ5LTIyIDQ5LTQ5VjEzN2MwLTI3LTIyLTQ5LTQ5LTQ5em0xNiA0OXYyMjZsLTIgNy0xMTUtMTE2IDExNy0xMTd6TTM2IDM2M1YxMzdsMTE3IDExN0wzOCAzNzBsLTItN3ptMjE5LTYzYy0zIDMtNyAzLTEwIDBMNjYgMTIxaDM2OHptLTc5LTIzIDQ2IDQ2YTM5IDM5IDAgMCAwIDU2IDBsNDYtNDYgMTAxIDEwMkg3NXoiIHN0eWxlPSJmaWxsOiNmNWY1ZjU7ZmlsbC1vcGFjaXR5OjEiLz48L3N2Zz4=" alt="Email"/></a>
<a href="https://twitter.com/GSFramework"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/GSFramework?style=social"></a>
</p>

<p align="center"><b>Youtube Tutorial on GSTools</b><br></p>

<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=qZBJ-AZXq6Q" target="_blank">
<img src="http://img.youtube.com/vi/qZBJ-AZXq6Q/0.jpg" alt="GSTools Transform 22 tutorial" width="480" height="360" border="0" />
</a>
</p>

## Purpose

<img align="right" width="450" src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/demonstrator.png" alt="">

GeoStatTools provides geostatistical tools for various purposes:
- random field generation
- simple, ordinary, universal and external drift kriging
- conditioned field generation
- incompressible random vector field generation
- (automated) variogram estimation and fitting
- directional variogram estimation and modelling
- data normalization and transformation
- many readily provided and even user-defined covariance models
- metric spatio-temporal modelling
- plotting and exporting routines


## Installation


### conda

GSTools can be installed via [conda][conda_link] on Linux, Mac, and Windows.
Install the package by typing the following command in a command terminal:

    conda install gstools

In case conda forge is not set up for your system yet, see the easy to follow
instructions on [conda forge][conda_forge_link]. Using conda, the parallelized
version of GSTools should be installed.


### pip

GSTools can be installed via [pip][pip_link] on Linux, Mac, and Windows.
On Windows you can install [WinPython][winpy_link] to get Python and pip
running. Install the package by typing the following command in a command terminal:

    pip install gstools

To install the latest development version via pip, see the
[documentation][doc_install_link].


## Citation

If you are using GSTools in your publication please cite our paper:

> Müller, S., Schüler, L., Zech, A., and Heße, F.:
> GSTools v1.3: a toolbox for geostatistical modelling in Python,
> Geosci. Model Dev., 15, 3161–3182, https://doi.org/10.5194/gmd-15-3161-2022, 2022.

You can cite the Zenodo code publication of GSTools by:

> Sebastian Müller & Lennart Schüler. GeoStat-Framework/GSTools. Zenodo. https://doi.org/10.5281/zenodo.1313628

If you want to cite a specific version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.1313628).


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
- [Geographic Coordinates][tut8_link]
- [Spatio-Temporal Modelling][tut9_link]
- [Normalizing Data][tut10_link]
- [Miscellaneous examples][tut0_link]

The associated python scripts are provided in the `examples` folder.


## Spatial Random Field Generation

The core of this library is the generation of spatial random fields. These fields are generated using the randomisation method, described by [Heße et al. 2014][rand_link].

[rand_link]: https://doi.org/10.1016/j.envsoft.2014.01.013


### Examples

#### Gaussian Covariance Model

This is an example of how to generate a 2 dimensional spatial random field with a gaussian covariance model.

```python
import gstools as gs
# structured field with a size 100x100 and a grid-size of 1x1
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model)
srf((x, y), mesh_type='structured')
srf.plot()
```
<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/gau_field.png" alt="Random field" width="600px"/>
</p>

GSTools also provides support for [geographic coordinates](https://en.wikipedia.org/wiki/Geographic_coordinate_system).
This works perfectly well with [cartopy](https://scitools.org.uk/cartopy/docs/latest/index.html).

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gstools as gs
# define a structured field by latitude and longitude
lat = lon = range(-80, 81)
model = gs.Gaussian(latlon=True, len_scale=777, geo_scale=gs.KM_SCALE)
srf = gs.SRF(model, seed=12345)
field = srf.structured((lat, lon))
# Orthographic plotting with cartopy
ax = plt.subplot(projection=ccrs.Orthographic(-45, 45))
cont = ax.contourf(lon, lat, field, transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
plt.colorbar(cont)
```

<p align="center">
<img src="https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_globe.png" alt="lat-lon random field" width="600px"/>
</p>

A similar example but for a three dimensional field is exported to a [VTK](https://vtk.org/) file, which can be visualized with [ParaView](https://www.paraview.org/) or [PyVista](https://docs.pyvista.org) in Python:

```python
import gstools as gs
# structured field with a size 100x100x100 and a grid-size of 1x1x1
x = y = z = range(100)
model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=(0.8, 0.4, 0.2))
srf = gs.SRF(model)
srf((x, y, z), mesh_type='structured')
srf.vtk_export('3d_field') # Save to a VTK file for ParaView

mesh = srf.to_pyvista() # Create a PyVista mesh for plotting in Python
mesh.contour(isosurfaces=8).plot()
```

<p align="center">
<img src="https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_pyvista.png" alt="3d Random field" width="600px"/>
</p>


## Estimating and Fitting Variograms

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.

All covariance models can be used to fit given variogram data by a simple interface.

### Example

This is an example of how to estimate the variogram of a 2 dimensional unstructured field and estimate the parameters of the covariance
model again.

```python
import numpy as np
import gstools as gs
# generate a synthetic field with an exponential model
x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = gs.Exponential(dim=2, var=2, len_scale=8)
srf = gs.SRF(model, mean=0, seed=19970221)
field = srf((x, y))
# estimate the variogram of the field
bin_center, gamma = gs.vario_estimate((x, y), field)
# fit the variogram with a stable model. (no nugget fitted)
fit_model = gs.Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)
# output
ax = fit_model.plot(x_max=max(bin_center))
ax.scatter(bin_center, gamma)
print(fit_model)
```

Which gives:

```python
Stable(dim=2, var=1.85, len_scale=7.42, nugget=0.0, anis=[1.0], angles=[0.0], alpha=1.09)
```

<p align="center">
<img src="https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/img/GS_vario_est.png" alt="Variogram" width="600px"/>
</p>


## Kriging and Conditioned Random Fields

An important part of geostatistics is Kriging and conditioning spatial random
fields to measurements. With conditioned random fields, an ensemble of field realizations with their variability depending on the proximity of the measurements can be generated.

### Example
For better visualization, we will condition a 1d field to a few "measurements", generate 100 realizations and plot them:

```python
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

# conditions
cond_pos = [0.3, 1.9, 1.1, 3.3, 4.7]
cond_val = [0.47, 0.56, 0.74, 1.47, 1.74]

# conditioned spatial random field class
model = gs.Gaussian(dim=1, var=0.5, len_scale=2)
krige = gs.krige.Ordinary(model, cond_pos, cond_val)
cond_srf = gs.CondSRF(krige)
# same output positions for all ensemble members
grid_pos = np.linspace(0.0, 15.0, 151)
cond_srf.set_pos(grid_pos)

# seeded ensemble generation
seed = gs.random.MasterRNG(20170519)
for i in range(100):
    field = cond_srf(seed=seed(), store=f"field_{i}")
    plt.plot(grid_pos, field, color="k", alpha=0.1)
plt.scatter(cond_pos, cond_val, color="k")
plt.show()
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/cond_ens.png" alt="Conditioned" width="600px"/>
</p>

## User Defined Covariance Models

One of the core-features of GSTools is the powerful
[CovModel][cov_link]
class, which allows to easy define covariance models by the user.

### Example

Here we re-implement the Gaussian covariance model by defining just a
[correlation][cor_link] function, which takes a non-dimensional distance ``h = r/l``:

```python
import numpy as np
import gstools as gs
# use CovModel as the base-class
class Gau(gs.CovModel):
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
import gstools as gs
x = np.arange(100)
y = np.arange(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, generator='VectorField', seed=19841203)
srf((x, y), mesh_type='structured')
srf.plot()
```

yielding

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/vec_srf_tut_gau.png" alt="vector field" width="600px"/>
</p>


[kraichnan_link]: https://doi.org/10.1063/1.1692799


## VTK/PyVista Export

After you have created a field, you may want to save it to file, so we provide
a handy [VTK][vtk_link] export routine using the `.vtk_export()` or you could
create a VTK/PyVista dataset for use in Python with to `.to_pyvista()` method:

```python
import gstools as gs
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model)
srf((x, y), mesh_type='structured')
srf.vtk_export("field") # Saves to a VTK file
mesh = srf.to_pyvista() # Create a VTK/PyVista dataset in memory
mesh.plot()
```

Which gives a RectilinearGrid VTK file ``field.vtr`` or creates a PyVista mesh
in memory for immediate 3D plotting in Python.

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/main/docs/source/pics/pyvista_export.png" alt="pyvista export" width="600px"/>
</p>


## Requirements:

- [NumPy >= 1.20.0](https://www.numpy.org)
- [SciPy >= 1.1.0](https://www.scipy.org/scipylib)
- [hankel >= 1.0.0](https://github.com/steven-murray/hankel)
- [emcee >= 3.0.0](https://github.com/dfm/emcee)
- [pyevtk >= 1.1.1](https://github.com/pyscience-projects/pyevtk)
- [meshio >= 5.1.0](https://github.com/nschloe/meshio)

### Optional

- [GSTools-Core >= 0.2.0](https://github.com/GeoStat-Framework/GSTools-Core)
- [matplotlib](https://matplotlib.org)
- [pyvista](https://docs.pyvista.org/)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[LGPLv3][license_link] © 2018-2024

[pip_link]: https://pypi.org/project/gstools
[conda_link]: https://docs.conda.io/en/latest/miniconda.html
[conda_forge_link]: https://github.com/conda-forge/gstools-feedstock#installing-gstools
[conda_pip]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages
[pipiflag]: https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i
[winpy_link]: https://winpython.github.io/
[license_link]: https://github.com/GeoStat-Framework/GSTools/blob/main/LICENSE
[cov_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/generated/gstools.covmodel.CovModel.html#gstools.covmodel.CovModel
[stable_link]: https://en.wikipedia.org/wiki/Stable_distribution
[doc_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/
[doc_install_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/#pip
[tut_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/tutorials.html
[tut1_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/01_random_field/index.html
[tut2_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/02_cov_model/index.html
[tut3_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/03_variogram/index.html
[tut4_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/04_vector_field/index.html
[tut5_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/05_kriging/index.html
[tut6_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/06_conditioned_fields/index.html
[tut7_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/07_transformations/index.html
[tut8_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/08_geo_coordinates/index.html
[tut9_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/09_spatio_temporal/index.html
[tut10_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/10_normalizer/index.html
[tut0_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/00_misc/index.html
[cor_link]: https://en.wikipedia.org/wiki/Autocovariance#Normalization
[vtk_link]: https://www.vtk.org/
