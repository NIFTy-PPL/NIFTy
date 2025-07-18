import nifty

needs_sphinx = "3.2.0"

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.mathjax",  # Render math as images
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Links to other sphinx docs (mostly numpy)
    "sphinx.ext.autodoc",
    "myst_parser",  # Parse markdown
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["user/paper.bib"]
master_doc = "index"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "strikethrough",
    "tasklist",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    # "matplotlib": ('https://matplotlib.org/stable/', None),
    "ducc0": ("https://mtr.pages.mpcdf.de/ducc/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

autodoc_default_options = {"special-members": "__init__"}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

imgmath_embed = True
numfig = True

project = "NIFTy"
copyright = "2013-2022, Max-Planck-Society"
author = "Martin Reinecke"

version = nifty.__version__[:-2]

language = "en"
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "light"}
html_logo = "nifty_logo_black.png"

html_theme_options = {
    "logo": {
        "image_light": "nifty_logo_black.png",
        "image_dark": "nifty_logo_black.png",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nifty",
            "icon": "fas fa-box",
        },
        {
            "name": "GitLab",
            "url": "https://gitlab.mpcdf.mpg.de/ift/nifty",
            "icon": "fab fa-gitlab",
        },
    ],
    "navbar_persistent": ["search-field"],
    "navbar_end": ["navbar-icon-links"],
}

html_last_updated_fmt = "%b %d, %Y"

exclude_patterns = ["mod/modules.rst", "mod/nifty.logger.rst"]
