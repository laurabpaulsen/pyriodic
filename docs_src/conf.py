# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # adjust if needed

project = 'pyriodic'
copyright = '2025, Laura Bock Paulsen'
author = 'Laura Bock Paulsen'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
]

html_theme = 'pydata_sphinx_theme'



html_theme_options = {
    "navbar_align": "content",
    "show_prev_next": False,
    "navigation_with_keys": True,
    "navbar_end": ["search-field.html", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/laurabpaulsen/pyriodic",
            "icon": "fab fa-github",
        },
    ],
}



#html_theme_options["navbar_links"] = [
#    {
#        "name": "Installation", 
#        "url": "installation/index", 
#        "internal": True,
#    },
#    {
#        "name": "Tutorials",
#         "url": "tutorials/index",
#          "internal": True,
#      },
#      {
#          "name": "API",
#          "url": "api/index",
#          "internal": True,
#      },
#]

html_static_path = ['source/_static']

autosummary_generate = True

