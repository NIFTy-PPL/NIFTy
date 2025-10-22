#!/usr/bin/sh

set -e

FOLDER=docs/source/user
FILES=" \
  nifty_cl_getting_started_0 \
  nifty_cl_getting_started_4_CorrelatedFields \
  nifty_cl_custom_nonlinearities \
  notebooks_re/0_models \
  notebooks_re/1_inference \
  notebooks_re/2_gaussian_processes \
  notebooks_re/3_wiener_filter \
  notebooks_re/4_correlated_field_model \
  notebooks_re/5_log_normal_poisson \
"

for FILE in $FILES; do
    if [ ! -f "${FILE}.md" ] || [ "${FILE}.ipynb" -nt "${FILE}.md" ]; then
        jupytext --to ipynb "${FOLDER}/${FILE}.py"
        jupyter-nbconvert --to markdown --execute --ExecutePreprocessor.timeout=None "${FOLDER}/${FILE}.ipynb"
    fi
done

jupyter-book build docs/source/user/notebooks_re --builder pdflatex
mkdir docs/source/_static
cp docs/source/user/notebooks_re/_build/latex/NIFTy_Introduction.pdf docs/source/_static

EXCLUDE="nifty/cl/logger.py, nifty/config.py"
sphinx-apidoc -e -d 1 -o docs/source/mod nifty ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
