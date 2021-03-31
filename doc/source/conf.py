import jifty1

extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',  # Render math as images
    'sphinx.ext.viewcode'  # Add links to highlighted source code
]
master_doc = 'index'

napoleon_google_docstring = False
napoleon_numpy_docstring = True

project = u'JIFTy7'
copyright = u'2013-2021, Max-Planck-Society'
author = u'Reimar Leike, Gordian Edenhofer'

release = jifty1.version.__version__
version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
# FIXME html_logo = "logo.png"

html_theme_options = {"gitlab_url": "https://gitlab.mpcdf.mpg.de/ift/jax_nifty"}
