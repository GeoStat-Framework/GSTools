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

- Fork the repo on [GitHub](https://github.com/GeoStat-Framework/GSTools) from the [develop branch](https://github.com/GeoStat-Framework/GSTools/tree/develop).
- Add yourself to AUTHORS.md (if you want to).
- We use the black code format, please use the script `black --line-length 79 gstools/` after you have written your code.
- Add some tests if possible.
- Add an example showing your new feature in one of the examples sub-folders if possible.
  Follow this [Sphinx-Gallary guide](https://sphinx-gallery.github.io/stable/syntax.html#embed-rst-in-your-example-python-files)
- Push to your fork and submit a pull request.
