# Welcome to GeoStatTools

[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://gstools.readthedocs.io/en/latest/)


## Purpose

GeoStatTools is a library providing geostatistical tools.


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

x = np.arange(0, 10, 120)
y = np.arange(-5, 5, 100)

cov_model = {'dim': 2, 'var': 1.6, 'len_scale': 4.5, 'model': 'gau', 'mode_no': 1000}

srf = SRF(**cov_model)
field = srf(x, y, seed=19970221, mesh_type='structured')
```


## License

[GPL][gpl_link] © 2018

[gpl_link]: https://github.com/LSchueler/GSTools/blob/master/LICENSE
