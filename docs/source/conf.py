# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pymgrid


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
    'sphinx.ext.doctest'
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


def autodoc_skip_member(app, what, name, obj, skip, options):
    try:
        doc = obj.__doc__
    except AttributeError:
        return None

    if doc is not None and ':meta private:' in doc:
        return True
    return None


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
