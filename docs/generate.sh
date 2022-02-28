set -e

FOLDER=docs/source/user/

for FILE in ${FOLDER}getting_started_0 ${FOLDER}getting_started_4_CorrelatedFields ${FOLDER}custom_nonlinearities
do
    if [ ! -f "${FILE}.rst" ] || [ ${FILE}.ipynb -nt ${FILE}.rst ]; then
		jupytext --to ipynb ${FILE}.py
        jupyter-nbconvert --to rst --execute --ExecutePreprocessor.timeout=None ${FILE}.ipynb
    fi
done

EXCLUDE="nifty8/logger.py nifty8/git_version.py"
sphinx-apidoc -e -o docs/source/mod nifty8 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
