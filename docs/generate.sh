jupyter-nbconvert --to rst --execute --ExecutePreprocessor.timeout=None docs/source/user/getting_started_0.ipynb
EXCLUDE="nifty7/logger.py nifty7/git_version.py"
sphinx-apidoc -e -o docs/source/mod nifty7 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
