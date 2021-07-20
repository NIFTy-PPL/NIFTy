jupyter-nbconvert --to rst --execute --ExecutePreprocessor.timeout=None docs/source/user/getting_started_0.ipynb

EXCLUDE="nifty8/logger.py nifty8/git_version.py"
sphinx-apidoc -e -o docs/source/mod nifty8 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
