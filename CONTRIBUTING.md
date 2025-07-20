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
  Please use the scripts `ruff check src/gstools`,
  `ruff check --select I --fix src/gstools/`, and
  `ruff format --diff src/gstools/` after you have written your code.
- Add some tests if possible.
- Add an example showing your new feature in one of the examples sub-folders if possible.
  Follow this [Sphinx-Gallary guide](https://sphinx-gallery.github.io/stable/syntax.html#embed-rst-in-your-example-python-files).
- Push to your fork and submit a pull request.

### PyLint Settings

Your code will be checked by [Pylint](https://github.com/PyCQA/pylint/)
with `pylint gstools` in the CI.
We made some generous default settings in `pyproject.toml` for the linter:

- max-args = 20
- max-locals = 50
- max-branches = 30
- max-statements = 80
- max-attributes = 25
- max-public-methods = 75

Since some classes in GSTools are quite huge and some function signatures are
somewhat longish.

By default [R0801](https://vald-phoenix.github.io/pylint-errors/plerr/errors/similarities/R0801)
(duplicate-code) is disabled, since it produces a lot of false positive errors
for docstrings and `__init__.py` settings.

We also disabled some pylint checks for some files by setting
comments like these at the beginning:
```python
# pylint: disable=C0103
```

Here is a list of the occurring disabled errors:
- [C0103](https://vald-phoenix.github.io/pylint-errors/plerr/errors/basic/C0103)
  (invalid-name) - `ax`, `r` etc. are marked as no valid names
- [C0302](https://vald-phoenix.github.io/pylint-errors/plerr/errors/format/C0302)
  (too-many-lines) - namely the `CovModel` definition has more than 1000 lines
- [C0415](https://vald-phoenix.github.io/pylint-errors/plerr/errors/imports/C0415)
  (import-outside-toplevel) - needed sometimes for deferred imports of optional
  dependencies like `matplotlib`
- [R0201](https://vald-phoenix.github.io/pylint-errors/plerr/errors/classes/R0201)
  (no-self-use) - methods with no `self` calls in some base-classes
- [W0212](https://vald-phoenix.github.io/pylint-errors/plerr/errors/classes/W0212)
  (protected-access) - we didn't want to draw attention to `CovModel._prec`
- [W0221](https://vald-phoenix.github.io/pylint-errors/plerr/errors/classes/W0221)
  (arguments-differ) - the `__call__` methods of `SRF` and `Krige` differ from `Field`
- [W0222](https://vald-phoenix.github.io/pylint-errors/plerr/errors/classes/W0222)
  (signature-differ) - the `__call__` methods of `SRF` and `Krige` differ from `Field`
- [W0231](https://vald-phoenix.github.io/pylint-errors/plerr/errors/classes/W0231)
  (super-init-not-called) - some child classes have their specialized `__init__`
- [W0613](https://vald-phoenix.github.io/pylint-errors/plerr/errors/variables/W0613)
  (unused-argument) - needed sometimes to match required call signatures
- [W0632](https://vald-phoenix.github.io/pylint-errors/plerr/errors/variables/W0632)
  (unbalanced-tuple-unpacking) - false positive for some call returns
- [E1101](https://vald-phoenix.github.io/pylint-errors/plerr/errors/typecheck/E1101)
  (no-member) - some times false positive
- [E1102](https://vald-phoenix.github.io/pylint-errors/plerr/errors/typecheck/E1102)
  (not-callable) - this is a false-positive result form some called properties
- [E1130](https://vald-phoenix.github.io/pylint-errors/plerr/errors/typecheck/E1130)
  (invalid-unary-operand-type) - false positive at some points

Although we disabled these errors at some points, we encourage you to prevent
disabling errors when it is possible.
