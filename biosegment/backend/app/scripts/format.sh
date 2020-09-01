#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place app --exclude .git,__pycache__,.env,.mypy_cache --per-file-ignores="__init__.py:F401"
black app
isort --recursive --apply app
