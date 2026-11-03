# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Path setup
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

import imatpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "iMATpy"
copyright = "2023, Braden Griebel"
author = "Braden Griebel"
release = imatpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_exec_code",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Change default settings -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
}
