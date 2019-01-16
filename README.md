# Welcome to GSTools

[![DOI](https://zenodo.org/badge/117534635.svg)](https://zenodo.org/badge/latestdoi/117534635)
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

GeoStatTools provides geostatistical tools for random field generation and variogram estimation based on many readily provided and even user-defined covariance models.


## Installation

The package can be installed via [pip][pip_link].
On Windows you can install [WinPython][winpy_link] to get
Python and pip running.

    pip install gstools


## Documentation for GSTools

You can find the documentation under [geostat-framework.readthedocs.io][doc_link].


### Tutorials and Examples

The documentation also includes some [tutorials][tut_link], showing the most important use cases of GSTools, which are

- [Random Field Generation][tut1_link]
- [The Covariance Model][tut2_link]
- [Variogram Estimation][tut3_link]

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
field = srf((x, y), mesh_type='structured')
plt.imshow(field)
plt.show()
```
<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/gau_field.png" alt="Random field" width="600px"/>
</p>

#### Truncated Power Law Model

GSTools also implements truncated power law variograms, which can be represented as a
superposition of scale dependant modes in form of standard variograms, which are truncated by
an upper lengthscale <i>l<sub>u</sub></i>.

This example shows the truncated power law based on the [stable model][stable_link] and is given by

<p align="center">
<img src="http://mathurl.com/yasd47ef.png" alt="Truncated Power Law - Stable"/>
</p>

which gives Gaussian modes for `alpha=2` or exponential modes for `alpha=1`

This results in:

<p align="center">
<img src="http://mathurl.com/yc669evd.png" alt="Truncated Power Law - Stable"/>
</p>

```python
import numpy as np
import matplotlib.pyplot as plt
from gstools import SRF, TPLStable
x = y = np.linspace(0, 100, 100)
model = TPLStable(
    dim=2,           # spatial dimension
    var=1,           # variance (C is calculated internally, so that the variance is actually 1)
    len_low=0,       # lower truncation of the power law
    len_scale=10,    # length scale (a.k.a. range), len_up = len_low + len_scale
    nugget=0.1,      # nugget
    anis=0.5,        # anisotropy between main direction and transversal ones
    angles=np.pi/4,  # rotation angles
    alpha=1.5,       # shape parameter from the stable model
    hurst=0.7,       # hurst coefficient from the power law
)
srf = SRF(model, mean=1, mode_no=1000, seed=19970221, verbose=True)
field = srf((x, y), mesh_type='structured', force_moments=True)
# show the field in xy coordinates
plt.imshow(field.T, origin="lower")
plt.show()
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/tplstable_field.png" alt="Random field" width="600px"/>
</p>


## Estimating and fitting variograms

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.

All covariance models can be used to fit given variogram data by a simple interface.

### Example

This is an example of how to estimate the variogram of a 2 dimensional unstructured field and estimate the parameters of the covariance
model again.

```python
import numpy as np
from gstools import SRF, Exponential, Stable, vario_estimate_unstructured
from gstools.covmodel.plot import plot_variogram
import matplotlib.pyplot as plt
# generate a synthetic field with an exponential model
x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = Exponential(dim=2, var=2, len_scale=8)
srf = SRF(model, mean=0, seed=19970221)
field = srf((x, y))
# estimate the variogram of the field with 40 bins
bins = np.arange(40)
bin_center, gamma = vario_estimate_unstructured((x, y), field, bins)
plt.plot(bin_center, gamma)
# fit the variogram with a stable model. (no nugget fitted)
fit_model = Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)
plot_variogram(fit_model, x_max=40)
# output
print(fit_model)
plt.show()
```

Which gives:

```python
Stable(dim=2, var=1.92, len_scale=8.15, nugget=0.0, anis=[1.], angles=[0.], alpha=1.05)
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/exp_vario_fit.png" alt="Variogram" width="600px"/>
</p>


## User defined covariance models

One of the core-features of GSTools is the powerfull
[CovModel][cov_link]
class, which allows to easy define covariance models by the user.

### Example

Here we reimplement the Gaussian covariance model by defining just the
[correlation][cor_link] function:

```python
from gstools import CovModel
import numpy as np
# use CovModel as the base-class
class Gau(CovModel):
    def correlation(self, r):
        return np.exp(-(r/self.len_scale)**2)
```

And that's it! With ``Gau`` you now have a fully working covariance model,
which you could use for field generation or variogram fitting as shown above.

Have a look at the [documentation ][doc_link] for further information on incorporating
optional parameters and optimizations.


## VTK Export

After you have created a field, you may want to save it to file, so we provide
a handy [VTK][vtk_link] export routine:

```python
from gstools import SRF, Gaussian, vtk_export
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
field = srf((x, y), mesh_type='structured')
vtk_export("field", (x, y), field, mesh_type='structured')
```

Which gives a RectilinearGrid VTK file ``field.vtr``.


## Requirements:

- [NumPy >= 1.8.2](https://www.numpy.org)
- [SciPy >= 0.19.1](https://www.scipy.org/scipylib)
- [hankel >= 0.3.6](https://github.com/steven-murray/hankel)
- [emcee](https://github.com/dfm/emcee)
- [pyevtk](https://bitbucket.org/pauloh/pyevtk)
- [six](https://github.com/benjaminp/six)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[GPL][gpl_link] © 2018-2019

[pip_link]: https://pypi.org/project/gstools
[winpy_link]: https://winpython.github.io/
[gpl_link]: https://github.com/GeoStat-Framework/GSTools/blob/master/LICENSE
[cov_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/covmodel.base.html#gstools.covmodel.base.CovModel
[stable_link]: https://en.wikipedia.org/wiki/Stable_distribution
[doc_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/
[tut_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorials.html
[tut1_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_01_srf.html
[tut2_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_02_cov.html
[tut3_link]: https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_03_vario.html
[cor_link]: https://en.wikipedia.org/wiki/Autocovariance#Normalization
[vtk_link]: https://www.vtk.org/
