#!/bin/bash

version=$(grep "Version:" DESCRIPTION |cut -f2 -d " ")
echo $version

# package
# no binary dist as C++
python3 setup.py sdist 
# upload
#twine upload --repository-url https://test.pypi.org/legacy/ dist/humanleague-$version.tar.gz
twine upload --repository-url https://upload.pypi.org/legacy/ dist/humanleague-$version.tar.gz

# # test package in tmp env
# # segregrated env PYTHONPATH="" to be certain
# virtualenv -p python3 --no-site-packages /tmp/env
# source /tmp/env/bin/activate

# # local wheel
# #python3 -m pip install  ~/dev/UKCensusAPI/dist/ukcensusapi-1.1.1-py3-none-any.whl
# # test pypi
# #python3 -m pip install --index-url https://test.pypi.org/simple/ UKCensusAPI
# # real pypi 
# python3 -m pip install humanleague

# # clean up
# deactivate
# rm -rf /tmp/env
