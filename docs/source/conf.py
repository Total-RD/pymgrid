# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pymgrid

import sys
sys.path.insert(0, pymgrid.PROJECT_PATH.parent)

print(f'Appended to path: {sys.path[0]}')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pymgrid'
copyright = '2022, TotalEnergies'
author = 'Avishai Halev'
release = pymgrid.__version__
version = pymgrid.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'nbsphinx',
    'nbsphinx_link',
    'IPython.sphinxext.ipython_console_highlighting'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"]
}

html_static_path = ['_static']

# -----------------------
# This block does not work

import inspect

def mask_docstrings(cls):
    if not cls.is_sink:
        cls.as_sink.__doc__ = '\t\t:meta private:\n' + cls.as_sink.__doc__
    if not cls.is_source:
        cls.as_source.__doc__ = '\t\t:meta private:\n' + cls.as_source.__doc__


for name, obj in inspect.getmembers(pymgrid.modules):
    break
    try:
        mask_docstrings(obj)
    except AttributeError:
        pass


# -----------------------

skip_members = ['yaml_flow_style']

def autodoc_skip_member(app, what, name, obj, skip, options):
    if name in skip_members:
        return True

    try:
        doc = obj.__doc__
    except AttributeError:
        return None

    if doc is not None and ':meta private:' in doc:
        return True
    return None


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
