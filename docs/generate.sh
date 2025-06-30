#!/usr/bin/sh

set -e

FOLDER=docs/source/user

for FILE in ${FOLDER}/0_intro ${FOLDER}/nifty_cl_getting_started_0 ${FOLDER}/nifty_cl_getting_started_4_CorrelatedFields ${FOLDER}/nifty_cl_custom_nonlinearities ${FOLDER}/a_correlated_field; do
    if [ ! -f "${FILE}.md" ] || [ "${FILE}.ipynb" -nt "${FILE}.md" ]; then
        jupytext --to ipynb "${FILE}.py"
        jupyter-nbconvert --to markdown --execute --ExecutePreprocessor.timeout=None "${FILE}.ipynb"
    fi
done

EXCLUDE="nifty/cl/logger.py, nifty/config.py"
sphinx-apidoc -e -d 1 -o docs/source/mod nifty ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
