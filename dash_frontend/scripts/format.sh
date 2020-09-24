#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place app --exclude .git,__pycache__,.env,.mypy_cache,__init__.py
black app
isort --recursive --apply app
