set -e

FOLDER=docs/source/user/

for FILE in ${FOLDER}0_wiener_filter ${FOLDER}1_inference_with_nifty ${FOLDER}2_jax_backend ${FOLDER}custom_nonlinearities ${FOLDER}custom_operators ${FOLDER}correlated_field_parameters
do
    if [ ! -f "${FILE}.rst" ] || [ ${FILE}.ipynb -nt ${FILE}.rst ]; then
		jupytext --to ipynb ${FILE}.py
        jupyter-nbconvert --to rst --execute --ExecutePreprocessor.timeout=None ${FILE}.ipynb
    fi
done

EXCLUDE="nifty8/logger.py"
sphinx-apidoc -e -o docs/source/mod nifty8 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
