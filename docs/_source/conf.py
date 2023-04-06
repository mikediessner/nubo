# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NUBO'
copyright = '2023, Mike Diessner'
author = 'Mike Diessner'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinxext.opengraph'
]

source_suffix = ['.rst', '.md']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_js_files = [
    'custom.js',
]

html_theme = 'furo'
html_title = 'NUBO'
pygments_style = "default"
pygments_dark_style = "monokai"
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
html_favicon = 'favicon.ico'
