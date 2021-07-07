Contributing to NIFTy
=====================


Coding conventions
------------------

We do not use pure Python `assert` statements in production code. They are not
guaranteed to by executed by Python and can be turned off by the user
(`python -O` in cPython). As an alternative use `ift.myassert`.


Build the documentation
-----------------------

To build the documentation from source, install `sphinx
<https://www.sphinx-doc.org/en/stable/index.html>`_ and the `pydata sphinx theme
<https://github.com/readthedocs/sphinx_rtd_theme>`_ on your system and run

.. code-block:: sh

   sh docs/generate.sh
