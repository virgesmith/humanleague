#!/bin/bash

. .env

version=$(grep "Version:" DESCRIPTION |cut -f2 -d " ")
echo $version

# package
# no binary dist as C++
python -m build . --sdist
# upload
# twine upload -u __token__ -p $TEST_PYPI_API_TOKEN  --repository testpypi dist/humanleague-$version.tar.gz
twine upload -u __token__ -p $PYPI_API_TOKEN dist/humanleague-$version.tar.gz

# NB on testPyPI, deps need to been installed from main repo. Use this:
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple humanleague
