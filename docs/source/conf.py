"""Sphinx configuration for AGILE documentation."""

project = "AGILE"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# Markdown support via MyST
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_extra_path = ["../../docs/videos", "../../docs/figures"]
