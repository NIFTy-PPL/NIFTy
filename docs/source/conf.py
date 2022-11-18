import nifty7

needs_sphinx = '3.2.0'

extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',  # Render math as images
    'sphinx.ext.viewcode'  # Add links to highlighted source code
]
master_doc = 'index'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

project = u'NIFTy7'
copyright = u'2013-2021, Max-Planck-Society'

imgmath_embed = True

author = u'Martin Reinecke'

release = nifty7.version.__version__
version = release[:-2]

language = "en"
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_logo = 'nifty_logo_black.png'

html_context = {
   "default_mode": "light"
}

html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nifty7",
            "icon": "fas fa-box",
        },
        {
            "name": "GitLab",
            "url": "https://gitlab.mpcdf.mpg.de/ift/nifty",
            "icon": "fab fa-gitlab",
        }
    ],
    "navbar_end": ["navbar-icon-links"]
}
html_last_updated_fmt = '%b %d, %Y'

exclude_patterns = [
    'mod/modules.rst', 'mod/nifty7.git_version.rst', 'mod/nifty7.logger.rst'
]
