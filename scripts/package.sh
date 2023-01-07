#!/bin/bash

. .env

version=$(grep "Version:" DESCRIPTION |cut -f2 -d " ")
echo $version

# package
# no binary dist as C++
python3 setup.py sdist
# upload
twine upload -u __token__ -p $TEST_PYPI_TOKEN  --repository-url https://test.pypi.org/legacy/ dist/humanleague-$version.tar.gz
# twine upload -u __token__ -p $PYPI_TOKEN --repository-url https://upload.pypi.org/legacy/ dist/humanleague-$version.tar.gz

# NB on testPyPI, deps need to been installed from main repo. Use this:
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple humanleague
