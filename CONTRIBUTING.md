# How to Contribute to GSTools

We are happy about all contributions! :thumbsup:


## Did you find a bug?

- Ensure that the bug was not already reported under
[GitHub issues](https://github.com/GeoStat-Framework/GSTools/issues)
- If the bug wasn't already reported, open a
[new issue](https://github.com/GeoStat-Framework/GSTools/issues) with a clear
description of the problem and if possible with a
[minimal working example](https://en.wikipedia.org/wiki/Minimal_working_example).
- please add the version number to the issue:

```python
import gstools
print(gstools.__version__)
```


## Do you have suggestions for new features?

Open a [new issue](https://github.com/GeoStat-Framework/GSTools/issues)
with your idea or suggestion and we'd love to discuss about it.


## Do you want to enhance GSTools or fix something?

- Fork the repo on [GitHub](https://github.com/GeoStat-Framework/GSTools)
- Add yourself to AUTHORS.md (if you want to).
- We use [Ruff](https://github.com/psf/black) to check and format the code.
  Please use the scripts `ruff check src/gstools` and
  `ruff format --diff src/gstools/` after you have written your code.
- Add some tests if possible.
- Add an example showing your new feature in one of the examples sub-folders if possible.
  Follow this [Sphinx-Gallary guide](https://sphinx-gallery.github.io/stable/syntax.html#embed-rst-in-your-example-python-files).
- Push to your fork and submit a pull request.


## Do you want to contribute to benchmarking?

GSTools tracks the runtime and memory performance 
using [Airspeed Velocity (ASV)](https://asv.readthedocs.io/).
The benchmark suite lives in the `benchmarks/` directory and covers both the
Cython fallback and the optional Rust backend, as well as OpenMP thread scaling.

Every pull request automatically runs the benchmark suite and reports any
significant regressions or improvements directly in the PR.
The cumulative benchmark history is published at
<https://geostat-framework.github.io/gstools-benchmarks/>.

If you would like to contribute to benchmarking you can:

- **Add a new benchmark** — follow the existing patterns in
  `benchmarks/` and prefix the function name with `time_` (runtime) or
  `peakmem_` (peak memory) so ASV picks it up automatically.
- **Improve the benchmark infrastructure** — the helper scripts in
  `benchmarks/tools/` generate the HTML comparison report and configure
  ASV for CI runs.
- **Report a performance regression** — open a
  [new issue](https://github.com/GeoStat-Framework/GSTools/issues) and
  include the relevant ASV comparison output.
