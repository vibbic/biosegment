#!/usr/bin/env bash

set -x
failure=false
# exit on every line so return code linter is 0
mypy app || failure=true
black app --check || failure=true
isort --recursive --check-only app || failure=true
flake8 --exclude .git,__pycache__,.env,.mypy_cache --per-file-ignores="__init__.py:F401" || failure=true

if [ "$failure" = true ]
then
    exit 1
else
    exit 0
fi