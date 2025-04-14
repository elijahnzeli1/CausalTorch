# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# !!! UPDATE THESE VALUES !!!
project = 'CausalTorch' # Replace with your project name
copyright = '2024, CausalTorch' # Replace with your copyright year and name
author = 'Nzeli Elijah' # Replace with your name
# The short X.Y version
# version = '0.1'
# The full version, including alpha/beta/rc tags
# release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    'sphinx.ext.napoleon', # Support for Google and NumPy style docstrings
    'sphinx.ext.intersphinx', # Link to other projects' documentation
    'sphinx.ext.viewcode', # Add links to source code
    # Add other extensions here if needed
]

# Configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
# Common themes: 'alabaster', 'sphinx_rtd_theme'
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Autodoc configuration ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class name.
autodoc_class_signature = "separated"

# -- Path setup for autodoc --------------------------------------------------
# If your project code is not directly installable, you might need to
# uncomment and adjust the following lines to tell Sphinx where to find your code.
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..')) # Point to the root of your project

print("---- sys.path for Sphinx ---")
# import sys
# print(sys.path)
print("---------------------------")
