# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

from copy import deepcopy
from builtins import object

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
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
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


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    """
    If a class signature is being read from cls.__new__, we want to replace it with the signature from cls.__init__.
    """
    if what == 'class' and signature[1:] in str(inspect.signature(obj.__new__)):
        obj_copy = deepcopy(obj)
        obj_copy.__new__ = object.__new__
        signature = str(inspect.signature(obj_copy))
        return signature, return_annotation


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(pymgrid.__file__))

    return f"https://github.com/Total-RD/pymgrid/tree/v{pymgrid.__version__}/src/pymgrid/{fn}{linespec}"


intersphinx_mapping = {
    'gym': ('https://www.gymlibrary.dev/', None)
}


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
    app.connect('autodoc-process-signature', autodoc_process_signature)
