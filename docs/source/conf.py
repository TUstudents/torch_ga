# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'torch_ga'
copyright = '2024, Francesco Alesiani'
author = 'Francesco Alesiani'

# The full version, including alpha/beta/rc tags
release = '0.0.6'
version = '0.0.6'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.intersphinx',   # Link to other projects' documentation
    'sphinx.ext.mathjax',       # Math support via MathJax
    'sphinx.ext.todo',          # Support for todo items
    'sphinx.ext.coverage',      # Check documentation coverage
    'myst_parser',              # Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for napoleon extension ------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for MyST parser -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",      # ::: can be used instead of ```
    "deflist",          # Definition lists 
    "dollarmath",       # $ and $$ for math
]

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True
