#!/usr/bin/env python3
#
# GSTools documentation build configuration file, created by
# sphinx-quickstart on Fri Jan  5 14:20:43 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# NOTE:
# pip install sphinx_rtd_theme
# is needed in order to build the documentation
# import os
# import sys
import datetime
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)

# local module should not be added to sys path if it's installed on RTFD
# see: https://stackoverflow.com/a/31882049/6696397
# sys.path.insert(0, os.path.abspath("../../"))
from gstools import __version__ as ver


def skip(app, what, name, obj, skip, options):
    if name in ["__call__"]:
        return False
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # parameters look better than with numpydoc only
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinxcontrib.youtube",
]

# autosummaries from source-files
autosummary_generate = True
# dont show __init__ docstring
autoclass_content = "class"
# sort class members
autodoc_member_order = "groupwise"
# autodoc_member_order = 'bysource'

# Notes in boxes
napoleon_use_admonition_for_notes = True
# Attributes like parameters
napoleon_use_ivar = True
# keep "Other Parameters" section
# https://github.com/sphinx-doc/sphinx/issues/10330
napoleon_use_param = False
# this is a nice class-doc layout
numpydoc_show_class_members = True
# class members have no separate file, so they are not in a toctree
numpydoc_class_members_toctree = False
# for the covmodels alot of classmembers show up...
# maybe switch off with:    :no-inherited-members:
numpydoc_show_inherited_class_members = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# source_suffix = [".rst", ".md"]
# source_suffix = ".rst"

# The master toctree document.
# --> this is the sitemap (or content-list in latex -> needs a heading)
# for html: the quickstart (in index.rst)
# gets the "index.html" and is therefore opened first
master_doc = "contents"

# General information about the project.
curr_year = datetime.datetime.now().year
project = "GSTools"
copyright = f"2018 - {curr_year}, Sebastian Müller, Lennart Schüler"
author = "Sebastian Müller, Lennart Schüler"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ver
# The full version, including alpha/beta/rc tags.
release = ver

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    #    'canonical_url': '',
    #    'analytics_id': '',
    "logo_only": False,
    "version_selector": True,
    "prev_next_buttons_location": "top",
    #    'style_external_links': False,
    #    'vcs_pageview_mode': '',
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 6,
    "includehidden": True,
    "titles_only": False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ["custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "GSToolsdoc"
# logos for the page
html_logo = "pics/gstools_150.png"
html_favicon = "pics/gstools.ico"

# -- Options for LaTeX output ---------------------------------------------
# latex_engine = 'lualatex'
# logo too big
latex_logo = "pics/gstools_150.png"

# latex_show_urls = 'footnote'
# http://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output
latex_elements = {
    "preamble": r"""
\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{2}
\pagestyle{fancy}
""",
    "pointsize": "10pt",
    "papersize": "a4paper",
    "fncychap": "\\usepackage[Glenn]{fncychap}",
    # 'inputenc': r'\usepackage[utf8]{inputenc}',
}
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "GSTools.tex",
        "GSTools Documentation",
        "Sebastian Müller, Lennart Schüler",
        "manual",
    )
]
# latex_use_parts = True

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "GSTools", "GSTools Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "GSTools",
        "GSTools Documentation",
        author,
        "GSTools",
        "Geo-statistical toolbox.",
        "Miscellaneous",
    )
]

suppress_warnings = [
    "image.nonlocal_uri",
    #    'app.add_directive',  # this evtl. suppresses the numpydoc induced warning
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "Python": ("https://docs.python.org/", None),
    "NumPy": ("https://numpy.org/doc/stable/", None),
    "SciPy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "hankel": ("https://hankel.readthedocs.io/en/latest/", None),
    "emcee": ("https://emcee.readthedocs.io/en/latest/", None),
}

# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey

# Use pyvista's image scraper for example gallery
# import pyvista
# https://github.com/tkoyama010/pyvista-doc-translations/blob/85c835a3ada3a2adefac06ba70e15a101ffa9162/conf.py#L21
# https://github.com/simpeg/discretize/blob/f414dd7ee7c5ba9a141cb2c37d4b71fdc531eae8/docs/conf.py#L334
# Make sure off screen is set to true when building locally
# pyvista.OFF_SCREEN = True
# # necessary when building the sphinx gallery
# pyvista.BUILDING_GALLERY = True
# # Optional - set parameters like theme or window size
# pyvista.set_plot_theme("document")

sphinx_gallery_conf = {
    # "image_scrapers": ("pyvista", "matplotlib"),
    "remove_config_comments": True,
    # only show "print" output as output
    "capture_repr": (),
    # path to your examples scripts
    "examples_dirs": [
        "../../examples/00_misc/",
        "../../examples/01_random_field/",
        "../../examples/02_cov_model/",
        "../../examples/03_variogram/",
        "../../examples/04_vector_field/",
        "../../examples/05_kriging/",
        "../../examples/06_conditioned_fields/",
        "../../examples/07_transformations/",
        "../../examples/08_geo_coordinates/",
        "../../examples/09_spatio_temporal/",
        "../../examples/10_normalizer/",
        "../../examples/11_plurigaussian/",
        "../../examples/12_sum_model/",
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "examples/00_misc/",
        "examples/01_random_field/",
        "examples/02_cov_model/",
        "examples/03_variogram/",
        "examples/04_vector_field/",
        "examples/05_kriging/",
        "examples/06_conditioned_fields/",
        "examples/07_transformations/",
        "examples/08_geo_coordinates/",
        "examples/09_spatio_temporal/",
        "examples/10_normalizer/",
        "examples/11_plurigaussian/",
        "examples/12_sum_model/",
    ],
    # Pattern to search for example files
    "filename_pattern": r"\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "examples/backreferences",
    # Modules for which function level galleries are created.  In
    "doc_module": "gstools",
    # "first_notebook_cell": (
    #     "%matplotlib inline\n"
    #     "from pyvista import set_plot_theme\n"
    #     "set_plot_theme('document')"
    # ),
    "matplotlib_animations": True,
}
