import jifty1

extensions = [
    'sphinx.ext.autodoc',  # Configure the order of methods
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',  # Render math as images
    'sphinx.ext.viewcode'  # Add links to highlighted source code
]
master_doc = 'index'

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True

project = 'JIFTy = JAX + NIFTy'
copyright = '2020-2021, Max-Planck-Society'
author = 'Gordian Edenhofer, Reimar Leike'

release = jifty1.version.__version__
version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
# FIXME html_logo = "logo.png"

html_theme_options = {"gitlab_url": "https://gitlab.mpcdf.mpg.de/ift/jax_nifty"}
