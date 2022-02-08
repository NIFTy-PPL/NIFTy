import nifty8

needs_sphinx = '3.2.0'

extensions = [
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',    # Render math as images
    'sphinx.ext.viewcode',   # Add links to highlighted source code
    'sphinx.ext.intersphinx' # Links to other sphinx docs (mostly numpy)
]
master_doc = 'index'

intersphinx_mapping = {"numpy": ("https://numpy.org/doc/stable/", None),
                       #"matplotlib": ('https://matplotlib.org/stable/', None),
                       "ducc0": ("https://mtr.pages.mpcdf.de/ducc/", None),
                       "scipy": ('https://docs.scipy.org/doc/scipy/reference/', None),
                       }

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

project = u'NIFTy8'
copyright = u'2013-2022, Max-Planck-Society'
author = u'Martin Reinecke'

release = nifty8.version.__version__
version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_logo = 'nifty_logo_black.png'
html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nifty8",
            "icon": "fas fa-box",
        }
    ],
    "gitlab_url": "https://gitlab.mpcdf.mpg.de/ift/nifty",
}
html_last_updated_fmt = '%b %d, %Y'

exclude_patterns = [
    'mod/modules.rst', 'mod/nifty8.git_version.rst', 'mod/nifty8.logger.rst'
]
