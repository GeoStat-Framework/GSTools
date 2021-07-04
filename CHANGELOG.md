# Changelog

All notable changes to **GSTools** will be documented in this file.

## [Unreleased]

### Bugfixes
- `vario_estimate` was altering the input field unter certain circumstances [#180](https://github.com/GeoStat-Framework/GSTools/issues/180)


## [1.3.1] - Pure Pink - 2021-06

### Enhancements
- Standalone use of Field class [#166](https://github.com/GeoStat-Framework/GSTools/issues/166)
- add social badges in README [#169](https://github.com/GeoStat-Framework/GSTools/issues/169), [#170](https://github.com/GeoStat-Framework/GSTools/issues/170)

### Bugfixes
- use `oldest-supported-numpy` to build cython extensions [#165](https://github.com/GeoStat-Framework/GSTools/pull/165)


## [1.3.0] - Pure Pink - 2021-04

### Topics

#### Geographical Coordinates Support ([#113](https://github.com/GeoStat-Framework/GSTools/issues/113))
- added boolean init parameter `latlon` to indicate a geographic model. When given, spatial dimension is fixed to `dim=3`, `anis` and `angles` will be ignored, since anisotropy is not well-defined on a sphere.
- add property `field_dim` to indicate the dimension of the resulting field. Will be 2 if `latlon=True`
- added yadrenko variogram, covariance and correlation method, since the geographic models are derived from standard models in 3D by plugging in the chordal distance of two points on a sphere derived from there great-circle distance `zeta`:
  - `vario_yadrenko`: given by `variogram(2 * np.sin(zeta / 2))`
  - `cov_yadrenko`: given by `covariance(2 * np.sin(zeta / 2))`
  - `cor_yadrenko`: given by `correlation(2 * np.sin(zeta / 2))`
- added plotting routines for yadrenko methods described above
- the `isometrize` and `anisometrize` methods will convert `latlon` tuples (given in degree) to points on the unit-sphere in 3D and vice versa
- representation of geographical models don't display the `dim`, `anis` and `angles` parameters, but `latlon=True`
- `fit_variogram` will expect an estimated variogram with great-circle distances given in radians
- **Variogram estimation**
  - `latlon` switch implemented in `estimate_vario` routine
  - will return a variogram estimated by the great-circle distance (haversine formula) given in radians
- **Field**
  - added plotting routines for latlon fields
  - no vector fields possible on latlon fields
  - corretly handle pos tuple for latlon fields

#### Krige Unification ([#97](https://github.com/GeoStat-Framework/GSTools/issues/97))
- Swiss Army Knife for kriging: The `Krige` class now provides everything in one place
- "Kriging the mean" is now possible with the switch `only_mean` in the call routine
- `Simple`/`Ordinary`/`Universal`/`ExtDrift`/`Detrended` are only shortcuts to `Krige` with limited input parameter list
- We now use the `covariance` function to build up the kriging matrix (instead of variogram)
- An `unbiased` switch was added to enable simple kriging (where the unbiased condition is not given)
- An `exact` switch was added to allow smother results, if a `nugget` is present in the model
- An `cond_err` parameter was added, where measurement error variances can be given for each conditional point
- pseudo-inverse matrix is now used to solve the kriging system (can be disabled by the new switch `pseudo_inv`), this is equal to solving the system with least-squares and prevents numerical errors
- added options `fit_normalizer` and `fit_variogram` to automatically fit normalizer and variogram to given data

#### Directional Variograms and Auto-binning ([#87](https://github.com/GeoStat-Framework/GSTools/issues/87), [#106](https://github.com/GeoStat-Framework/GSTools/issues/106), [#131](https://github.com/GeoStat-Framework/GSTools/issues/131))
- new routine name `vario_estimate` instead of `vario_estimate_unstructured` (old kept for legacy code) for simplicity
- new routine name `vario_estimate_axis` instead of `vario_estimate_structured` (old kept for legacy code) for simplicity
- **vario_estimate**
  - added simple automatic binning routine to determine bins from given data (one third of box diameter as max bin distance, sturges rule for number of bins)
  - allow to pass multiple fields for joint variogram estimation (e.g. for daily precipitation) on same mesh
  - `no_data` option added to allow missing values
  - **masked fields**
    - user can now pass a masked array (or a list of masked arrays) to deselect data points.
    - in addition, a `mask` keyword was added to provide an external mask
  - **directional variograms**
    - diretional variograms can now be estimated
    - either provide a list of direction vectors or angles for directions (spherical coordinates)
    - can be controlled by given angle tolerance and (optional) bandwidth
    - prepared for nD
  - structured fields (pos tuple describes axes) can now be passed to estimate an isotropic or directional variogram
  - distance calculation in cython routines in now independent of dimension
- **vario_estimate_axis**
  - estimation along array axis now possible in arbitrary dimensions
  - `no_data` option added to allow missing values (sovles [#83](https://github.com/GeoStat-Framework/GSTools/issues/83))
  - axis can be given by name (`"x"`, `"y"`, `"z"`) or axis number (`0`, `1`, `2`, `3`, ...)

#### Better Variogram fitting ([#78](https://github.com/GeoStat-Framework/GSTools/issues/78), [#145](https://github.com/GeoStat-Framework/GSTools/pull/145))
- fixing sill possible now
- `loss` is now selectable for smoother handling of outliers
- r2 score can now be returned to get an impression of the goodness of fitting
- weights can be passed
- instead of deselecting parameters, one can also give fix values for each parameter
- default init guess for `len_scale` is now mean of given bin-centers
- default init guess for `var` and `nugget` is now mean of given variogram values

#### CovModel update ([#109](https://github.com/GeoStat-Framework/GSTools/issues/109), [#122](https://github.com/GeoStat-Framework/GSTools/issues/122), [#157](https://github.com/GeoStat-Framework/GSTools/pull/157))
- add new `rescale` argument and attribute to the `CovModel` class to be able to rescale the `len_scale` (usefull for unit conversion or rescaling `len_scale` to coincide with the `integral_scale` like it's the case with the Gaussian model)
  See: [#90](https://github.com/GeoStat-Framework/GSTools/issues/90), [GeoStat-Framework/PyKrige#119](https://github.com/GeoStat-Framework/PyKrige/issues/119)
- added new `len_rescaled` attribute to the `CovModel` class, which is the rescaled `len_scale`: `len_rescaled = len_scale / rescale`
- new method `default_rescale` to provide default rescale factor (can be overridden)
- remove `doctest` calls
- docstring updates in CovModel and derived models
- updated all models to use the `cor` routine and make use of the `rescale` argument (See: [#90](https://github.com/GeoStat-Framework/GSTools/issues/90))
- TPL models got a separate base class to not repeat code
- added **new models** (See: [#88](https://github.com/GeoStat-Framework/GSTools/issues/88)):
  -  `HyperSpherical`: (Replaces the old `Intersection` model) Derived from the intersection of hyper-spheres in arbitrary dimensions. Coincides with the linear model in 1D, the circular model in 2D and the classical spherical model in 3D
  - `SuperSpherical`: like the HyperSpherical, but the shape parameter derived from dimension can be set by the user. Coincides with the HyperSpherical model by default
  - `JBessel`: a hole model valid in all dimensions. The shape parameter controls the dimension it was derived from. For `nu=0.5` this model coincides with the well known `wave` hole model.
  - `TPLSimple`: a simple truncated power law controlled by a shape parameter `nu`. Coincides with the truncated linear model for `nu=1`
  - `Cubic`: to be compatible with scikit-gstat in the future
- all arguments are now stored as float internally ([#157](https://github.com/GeoStat-Framework/GSTools/pull/157))
- string representation of the `CovModel` class is now using a float precision (`CovModel._prec=3`) to truncate longish output
- string representation of the `CovModel` class now only shows `anis` and `angles` if model is anisotropic resp. rotated
- dimension validity check: raise a warning, if given model is not valid in the desired dimension (See: [#86](https://github.com/GeoStat-Framework/GSTools/issues/86))

#### Normalizer, Trend and Mean ([#124](https://github.com/GeoStat-Framework/GSTools/issues/124))

- new `normalize` submodule containing power-transforms for data to gain normality
- Base-Class: `Normalizer` providing basic functionality including maximum likelihood fitting
- added: `LogNormal`, `BoxCox`, `BoxCoxShift`, `YeoJohnson`, `Modulus` and `Manly`
- normalizer, trend and mean can be passed to SRF, Krige and variogram estimation routines
  - A trend can be a callable function, that represents a trend in input data. For example a linear decrease of temperature with height.
  - The normalizer will be applied after the data was detrended, i.e. the trend was substracted from the data, in order to gain normality.
  - The mean is now interpreted as the mean of the normalized data. The user could also provide a callable mean, but it is mostly meant to be constant.

#### Arbitrary dimensions ([#112](https://github.com/GeoStat-Framework/GSTools/issues/112))
- allow arbitrary dimensions in all routines (CovModel, Krige, SRF, variogram)
- anisotropy and rotation following a generalization of tait-bryan angles
- CovModel provides `isometrize` and `anisometrize` routines to convert points

#### New Class for Conditioned Random Fields ([#130](https://github.com/GeoStat-Framework/GSTools/issues/130))
- **THIS BREAKS BACKWARD COMPATIBILITY**
- `CondSRF` replaces the conditioning feature of the SRF class, which was cumbersome and limited to Ordinary and Simple kriging
- `CondSRF` behaves similar to the `SRF` class, but instead of a covariance model, it takes a kriging class as input. With this kriging class, all conditioning related settings are defined.

### Enhancements
- Python 3.9 Support [#107](https://github.com/GeoStat-Framework/GSTools/issues/107)
- add routines to format struct. pos tuple by given `dim` or `shape`
- add routine to format struct. pos tuple by given `shape` (variogram helper)
- remove `field.tools` subpackage
- support `meshio>=4.0` and add as dependency
- PyVista mesh support [#59](https://github.com/GeoStat-Framework/GSTools/issues/59)
- added `EARTH_RADIUS` as constant providing earths radius in km (can be used to rescale models)
- add routines `latlon2pos` and `pos2latlon` to convert lat-lon coordinates to points on unit-sphere and vice versa
- a lot of new examples and tutorials
- `RandMeth` class got a switch to select the sampling strategy
- plotter for n-D fields added [#141](https://github.com/GeoStat-Framework/GSTools/issues/141)
- antialias for contour plots of 2D fields [#141](https://github.com/GeoStat-Framework/GSTools/issues/141)
- building from source is now configured with `pyproject.toml` to care about build dependencies, see [#154](https://github.com/GeoStat-Framework/GSTools/issues/154)

### Changes
- drop support for Python 3.5 [#146](https://github.com/GeoStat-Framework/GSTools/pull/146)
- added a finit limit for shape-parameters in some CovModels [#147](https://github.com/GeoStat-Framework/GSTools/pull/147)
- drop usage of `pos2xyz` and `xyz2pos`
- remove structured option from generators (structured pos need to be converted first)
- explicitly assert dim=2,3 when generating vector fields
- simplify `pre_pos` routine to save pos tuple and reformat it an unstructured tuple
- simplify field shaping
- simplify plotting routines
- only the `"unstructured"` keyword is recognized everywhere, everything else is interpreted as `"structured"` (e.g. `"rectilinear"`)
- use GitHub-Actions instead of TravisCI
- parallel build now controlled by env-var `GSTOOLS_BUILD_PARALLEL=1`, see [#154](https://github.com/GeoStat-Framework/GSTools/issues/154)
- install extra target for `[dev]` dropped, can be reproduced by `pip install gstools[test, doc]`, see [#154](https://github.com/GeoStat-Framework/GSTools/issues/154)

### Bugfixes
- typo in keyword argument for vario_estimate_structured [#80](https://github.com/GeoStat-Framework/GSTools/issues/80)
- isotropic rotation of SRF was not possible [#100](https://github.com/GeoStat-Framework/GSTools/issues/100)
- `CovModel.opt_arg` now sorted [#103](https://github.com/GeoStat-Framework/GSTools/issues/103)
- CovModel.fit: check if weights are given as a string (numpy comparison error) [#111](https://github.com/GeoStat-Framework/GSTools/issues/111)
- several pylint fixes ([#159](https://github.com/GeoStat-Framework/GSTools/pull/159))

## [1.2.1] - Volatile Violet - 2020-04-14

### Bugfixes
- `ModuleNotFoundError` is not present in py35
- Fixing Cressie-Bug #76
- Adding analytical formula for integral scales of rational and stable model
- remove prange from IncomprRandMeth summators to prevent errors on Win and macOS


## [1.2.0] - Volatile Violet - 2020-03-20

### Enhancements
- different variogram estimator functions can now be used #51
- the TPLGaussian and TPLExponential now have analytical spectra #67
- added property ``is_isotropic`` to CovModel #67
- reworked the whole krige sub-module to provide multiple kriging methods #67
  - Simple
  - Ordinary
  - Universal
  - External Drift Kriging
  - Detrended Kriging
- a new transformation function for discrete fields has been added #70
- reworked tutorial section in the documentation #63
- pyvista interface #29

### Changes
- Python versions 2.7 and 3.4 are no longer supported #40 #43
- CovModel: in 3D the input of anisotropy is now treated slightly different: #67
  - single given anisotropy value [e] is converted to [1, e] (it was [e, e] before)
  - two given length-scales [l_1, l_2] are converted to [l_1, l_2, l_2] (it was [l_1, l_2, l_1] before)

### Bugfixes
- a race condition in the structured variogram estimation has been fixed #51


## [1.1.1] - Reverberating Red - 2019-11-08

### Enhancements
- added a changelog. See: [commit fbea883](https://github.com/GeoStat-Framework/GSTools/commit/fbea88300d0862393e52f4b7c3d2b15c2039498b)

### Changes
- deprecation warnings are now printed if Python versions 2.7 or 3.4 are used #40 #41

### Bugfixes
- define spectral_density instead of spectrum in covariance models since Cov-base derives spectrum. See: [commit 00f2747](https://github.com/GeoStat-Framework/GSTools/commit/00f2747fd0503ff8806f2eebfba36acff813416b)
- better boundaries for CovModel parameters. See: https://github.com/GeoStat-Framework/GSTools/issues/37


## [1.1.0] - Reverberating Red - 2019-10-01

### Enhancements
- by using Cython for all the heavy computations, we could achieve quite some speed ups and reduce the memory consumption significantly #16
- parallel computation in Cython is now supported with the help of OpenMP and the performance increase is nearly linear with increasing cores #16
- new submodule ``krige`` providing simple (known mean) and ordinary (estimated mean) kriging working analogous to the srf class
- interface to pykrige to use the gstools CovModel with the pykrige routines (https://github.com/bsmurphy/PyKrige/issues/124)
- the srf class now provides a ``plot`` and a ``vtk_export`` routine
- incompressible flow fields can now be generated #14
- new submodule providing several field transformations like: Zinn&Harvey, log-normal, bimodal, ... #13
- Python 3.4 and 3.7 wheel support #19
- field can now be generated directly on meshes from [meshio](https://github.com/nschloe/meshio) and [ogs5py](https://github.com/GeoStat-Framework/ogs5py), see: [commit f4a3439](https://github.com/GeoStat-Framework/GSTools/commit/f4a3439400b81d8d9db81a5f7fbf6435f603cf05)
- the srf and kriging classes now store the last ``pos``, ``mesh_type`` and ``field`` values to keep them accessible, see: [commit 29f7f1b](https://github.com/GeoStat-Framework/GSTools/commit/29f7f1b029866379ce881f44765f72534d757fae)
- tutorials on all important features of GSTools have been written for you guys #20
- a new interface to pyvista is provided to export fields to python vtk representation, which can be used for plotting, exploring and exporting fields #29

### Changes
- the license was changed from GPL to LGPL in order to promote the use of this library #25
- the rotation angles are now interpreted in positive direction (counter clock wise)
- the ``force_moments`` keyword was removed from the SRF call method, it is now in provided as a field transformation #13
- drop support of python implementations of the variogram estimators #18
- the ``variogram_normed`` method was removed from the ``CovModel`` class due to redundance [commit 25b1647](https://github.com/GeoStat-Framework/GSTools/commit/25b164722ac6744ebc7e03f3c0bf1c30be1eba89)
- the position vector of 1D fields does not have to be provided in a list-like object with length 1 [commit a6f5be8](https://github.com/GeoStat-Framework/GSTools/commit/a6f5be8bfd2db1f002e7889ecb8e9a037ea08886)

### Bugfixes
- several minor bugfixes


## [1.0.1] - Bouncy Blue - 2019-01-18

### Bugfixes
- fixed Numpy and Cython version during build process


## [1.0.0] - Bouncy Blue - 2019-01-16

### Enhancements
- added a new covariance class, which allows the easy usage of arbitrary covariance models
- added many predefined covariance models, including truncated power law models
- added [tutorials](https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorials.html) and examples, showing and explaining the main features of GSTools
- variogram models can be fitted to data
- prebuilt binaries for many Linux distributions, Mac OS and Windows, making the installation, especially of the Cython code, much easier
- the generated fields can now easily be exported to vtk files
- variance scaling is supported for coarser grids
- added pure Python versions of the variogram estimators, in case somebody has problems compiling Cython code
- the [documentation](https://geostat-framework.readthedocs.io/projects/gstools/en/latest/) is now a lot cleaner and easier to use
- the code is a lot cleaner and more consistent now
- unit tests are now automatically tested when new code is pushed
- test coverage of code is shown
- GeoStat Framework now has a website, visit us: https://geostat-framework.github.io/

### Changes
- release is not downwards compatible with release v0.4.0
- SRF creation has been adapted for the CovModel
- a tuple `pos` is now used instead of `x`, `y`, and `z` for the axes
- renamed `estimate_unstructured` and `estimate_structured` to `vario_estimate_unstructured` and `vario_estimate_structured` for less ambiguity

### Bugfixes
- several minor bugfixes


## [0.4.0] - Glorious Green - 2018-07-17

### Bugfixes
- import of cython functions put into a try-block


## [0.3.6] - Original Orange - 2018-07-17

First release of GSTools.


[Unreleased]: https://github.com/GeoStat-Framework/gstools/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GeoStat-Framework/gstools/compare/0.4.0...v1.0.0
[0.4.0]: https://github.com/GeoStat-Framework/gstools/compare/0.3.6...0.4.0
[0.3.6]: https://github.com/GeoStat-Framework/gstools/releases/tag/0.3.6
