set -e

FOLDER=docs/source/user/

for FILE in ${FOLDER}/nifty_re ${FOLDER}/old_nifty_getting_started_0 ${FOLDER}/old_nifty_getting_started_4_CorrelatedFields ${FOLDER}/old_nifty_custom_nonlinearities; do
    if [[ ! -f "${FILE}.rst" ]] || [ ${FILE}.ipynb -nt ${FILE}.rst ]; then
		jupytext --to ipynb ${FILE}.py
        jupyter-nbconvert --to rst --execute --ExecutePreprocessor.timeout=None ${FILE}.ipynb
    fi
done

EXCLUDE="nifty8/logger.py"
sphinx-apidoc -e -o docs/source/mod nifty8 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
