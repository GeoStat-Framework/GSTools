# Welcome to GeoStatTools

[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://gstools.readthedocs.io/en/latest/)
<p align="center">
<img src="/docs/source/gstools.png" alt="GSTools-LOGO" width="251px"/>
</p>
## Purpose

GeoStatTools is a library providing geostatistical tools.


## Installation

##### Requirements:
- numpy
- scipy

##### Installation:
`pip install gstools`


## Documentation for GeoStatTools

You can find the documentation [here][doc_link].

[doc_link]: https://gstools.readthedocs.io/en/latest/


## Spatial Random Field Generation

The core of this library is the generation of spatial random fields.


### Example

This is an example of how to generate a 2 dimensional spatial random field with a Gaussian covariance structure.

```python
import numpy as np
from gstools.field import SRF

x = np.linspace(0, 10, 120)
y = np.linspace(-5, 5, 100)

cov_model = {'dim': 2, 'var': 1.6, 'len_scale': 4.5, 'model': 'gau', 'mode_no': 1000}

srf = SRF(**cov_model)
field = srf(x, y, seed=19970221, mesh_type='structured')
```


## Estimating variograms

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.


### Example

This is an example of how to estimate the variogram of a 2 dimensional unstructured field.

```python
import numpy as np
from gstools.field import SRF
from gstools import variogram

#random samples between 0 <= x, y < 100
x = np.random.rand(1000) * 100.
y = np.random.rand(1000) * 100.

srf = SRF(dim=2, var=2, len_scale=30)
field = srf(x, y, seed=20011012)

bins = np.arange(0, 50)

gamma = variogram.estimate_unstructured(field, bins, x, y)
```


## License

[GPL][gpl_link] © 2018

[gpl_link]: https://github.com/LSchueler/GSTools/blob/master/LICENSE
