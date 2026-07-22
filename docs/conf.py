# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import os
from pathlib import Path

from packaging.version import Version

# -- Read the Docs Specific Configuration --------------------------------------

# This needs to be done before anything is imported
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    os.environ["SUNPY_CONFIGDIR"] = "/home/docs/"
    os.environ["HOME"] = "/home/docs/"
    os.environ["LANG"] = "C"
    os.environ["LC_ALL"] = "C"
    os.environ["PARFIVE_HIDE_PROGRESS"] = "True"

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from muse import __version__

_version = Version(__version__)
version = release = str(_version)
# Avoid "post" appearing in version string in rendered docs
if _version.is_postrelease:
    version = release = _version.base_version
# Avoid long githashes in rendered Sphinx docs
elif _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
is_development = _version.is_devrelease
is_release = not (_version.is_prerelease or _version.is_devrelease)

project = "muse"
author = "MUSE Instrument Team @ LMSAL"
copyright = f"{datetime.datetime.now(datetime.UTC).year}, {author}"  # NOQA: A001

# -- General configuration ---------------------------------------------------

# Wrap large function/method signatures
maximum_signature_line_length = 80

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_changelog",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxext.opengraph",
    "sphinx_autodoc_typehints",
]

# Register the template for the robots.txt
html_extra_path = ["robots.txt"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# Enable and configure nitpicky mode
nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
with Path("nitpick-exceptions").open() as nitpick_exceptions:
    for line in nitpick_exceptions:
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "https://www.astropy.org/astropy-data/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "https://www.astropy.org/astropy-data/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "https://www.astropy.org/astropy-data/intersphinx/scipy.inv"),
    ),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "sunpy": ("https://docs.sunpy.org/en/stable/", None),
    "sunkit_instruments": (
        "https://docs.sunpy.org/projects/sunkit-instruments/en/stable/",
        None,
    ),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
}

# -- Options for sphinxext-opengraph ------------------------------------------

ogp_image = "https://raw.githubusercontent.com/sunpy/sunpy-logo/master/generated/sunpy_logo_word.png"
ogp_use_first_image = True
ogp_description_length = 160
ogp_custom_meta_tags = ('<meta property="og:ignore_canonical" content="true" />',)

# -- Options for sphinx-copybutton ---------------------------------------------

# Python Repl + continuation, Bash, ipython and qtconsole + continuation, jupyter-console + continuation
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Keep the header navbar on one line: short brand text instead of
# "muse <version> documentation", and collapse extra links into "More" sooner.
# Header link order follows the toctree order in index.rst. The version string
# stays visible (browser tab and next to the logo) so release vs dev is clear.
html_title = f"muse {version}"
html_theme_options = {
    "header_links_before_dropdown": 4,
}
if is_development:
    html_theme_options["announcement"] = (
        "This is documentation for an unreleased development version of <code>muse</code>."
    )

# Drop the mission logo at docs/_static/muse_logo.png (or .svg, adjusting the
# name below) and it is picked up automatically; the theme's CSS scales it to
# the header height, so no manual rescaling is needed.
html_static_path = ["_static"]
_logo = Path(__file__).parent / "_static" / "muse_logo.png"
if _logo.exists():
    html_theme_options["logo"] = {
        "text": version,
        "image_light": f"_static/{_logo.name}",
        "image_dark": f"_static/{_logo.name}",
    }

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]


# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"


# -- Other options ----------------------------------------------------------

gallery_mode = os.environ.get("MUSE_GALLERY_MODE", "unskipped")
sphinx_gallery_conf = {
    "backreferences_dir": str(Path("generated") / "modules"),
    "filename_pattern": ".*" if gallery_mode == "all" else "^((?!skip_).)*$",
    "examples_dirs": str(Path("..") / "examples"),
    "within_subsection_order": "ExampleTitleSortKey",
    "gallery_dirs": str(Path("generated") / "gallery"),
    "abort_on_example_error": False,
    "plot_gallery": gallery_mode != "none",
    "remove_config_comments": True,
    "doc_module": ("muse"),
    "only_warn_on_example_error": True,
    "matplotlib_animations": True,
    "show_memory": True,
}


# -- Custom autodoc rendering ------------------------------------------------
# Defaults with lots of values are a pain.

# The docs/api pages are generated by automodapi, so per-object autodoc options
# like :no-value: cannot live in checked-in rst. Instead, any module-level
# variable can opt out of its ``= <value>`` annotation by putting
# ``:meta hide-value:`` in its docstring (DEFAULTS_MUSE does this).


def _append_defaults_repr(app, what, name, obj, options, lines):  # NOQA: ARG001
    # Sphinx collapses newlines in ``autodata`` value annotations, so we append
    # the itemized repr as a readable code block instead.
    if name == "muse.variables.DEFAULTS_MUSE":
        from muse.variables import DEFAULTS_MUSE  # NOQA: PLC0415

        lines.extend(["", ".. code-block:: text", ""])
        lines.extend(f"    {line}" for line in repr(DEFAULTS_MUSE).splitlines())
        lines.append("")


def setup(app):
    app.connect("autodoc-process-docstring", _append_defaults_repr)
