rm -rf docs/build docs/source/mod
EXCLUDE="nifty6/logger.py nifty6/git_version.py"
sphinx-apidoc -e -o docs/source/mod nifty6 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
