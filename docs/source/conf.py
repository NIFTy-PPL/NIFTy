import nifty5
import sphinx_rtd_theme

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_keyword = False

autodoc_member_order = 'groupwise'
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

extensions = [
    'sphinx.ext.autodoc', 'numpydoc', 'sphinx.ext.autosummary',
    'sphinx.ext.napoleon', 'sphinx.ext.imgmath', 'sphinx.ext.viewcode'
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'NIFTy5'
copyright = u'2013-2018, Max-Planck-Society'
author = u'Martin Reinecke'

release = nifty5.version.__version__
version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False
pygments_style = 'sphinx'
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'collapse_navigation': False,
    'display_version': False,
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = 'nifty_logo_black.png'
html_static_path = []
html_last_updated_fmt = '%b %d, %Y'
html_domain_indices = False
html_use_index = False
html_show_sourcelink = False
htmlhelp_basename = 'NIFTydoc'
