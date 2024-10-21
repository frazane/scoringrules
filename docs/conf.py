# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "scoringrules"
author = "scoringrules contributors"
copyright = "2024"
release = "0.7.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = [
    "image.not_readable",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/frazane/scoringrules",
    "use_repository_button": True,
    "show_toc_level": 3,
    "logo": {
        "image_light": "_static/banner_light.svg",
        "image_dark": "_static/banner_dark.svg",
    },
    "pygments_light_style": "lovelace",
    "pygments_dark_style": "monokai",
}
html_title = "scoringrules"
html_favicon = "_static/favicon.ico"
