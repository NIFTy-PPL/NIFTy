find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} \; 2> /dev/null
rm -rf build dist oprofile_data
rm -rf *.egg-info .eggs
rm -f log.log*
rm -f *.pdf *.png
rm -rf nifty2go
find . -type d -empty -delete
