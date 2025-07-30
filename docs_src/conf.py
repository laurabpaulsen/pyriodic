# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyriodic'
copyright = '2025, Laura Bock Paulsen'
author = 'Laura Bock Paulsen'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

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
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/laurabpaulsen/pyriodic",
            "icon": "fab fa-github",
        },
    ],
    "navigation_with_keys": True
    }



html_theme_options["navbar_links"] = [
    {"name": "Installation", "url": "installation", "internal": True},
    {
        "name": "Tutorials",
        "url": "tutorials/index",
        "internal": True,
    },
    {
        "name": "API",
        "url": "api/index",
        "internal": True,
    },
]

html_static_path = ['source/_static']

autosummary_generate = True

