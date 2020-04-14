# Changelog

All notable changes to **GSTools** will be documented in this file.

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
- added a changelog. See: https://github.com/GeoStat-Framework/GSTools/commit/fbea88300d0862393e52f4b7c3d2b15c2039498b

### Changes
- deprecation warnings are now printed if Python versions 2.7 or 3.4 are used #40 #41

### Bugfixes
- define spectral_density instead of spectrum in covariance models since Cov-base derives spectrum. See: https://github.com/GeoStat-Framework/GSTools/commit/00f2747fd0503ff8806f2eebfba36acff813416b
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
- field can now be generated directly on meshes from [``meshio``](https://github.com/nschloe/meshio) and [``ogs5py``](https://github.com/GeoStat-Framework/ogs5py) f4a3439400b8
- the srf and kriging classes now store the last ``pos``, ``mesh_type`` and ``field`` values to keep them accessible 29f7f1b02
- tutorials on all important features of GSTools have been written for you guys #20
- a new interface to pyvista is provided to export fields to python vtk representation, which can be used for plotting, exploring and exporting fields #29

### Changes
- the license was changed from GPL to LGPL in order to promote the use of this library #25
- the rotation angles are now interpreted in positive direction (counter clock wise)
- the ``force_moments`` keyword was removed from the SRF call method, it is now in provided as a field transformation #13
- drop support of python implementations of the variogram estimators #18
- the ``variogram_normed`` method was removed from the ``CovModel`` class due to redundance 25b164722ac6744ebc7e03f3c0bf1c30be1eba89
- the position vector of 1D fields does not have to be provided in a list-like object with length 1 a6f5be8bf

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


[Unreleased]: https://github.com/GeoStat-Framework/gstools/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GeoStat-Framework/gstools/compare/0.4.0...v1.0.0
[0.4.0]: https://github.com/GeoStat-Framework/gstools/compare/0.3.6...0.4.0
[0.3.6]: https://github.com/GeoStat-Framework/gstools/releases/tag/0.3.6
