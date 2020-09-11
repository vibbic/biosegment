#!/usr/bin/env bash

set -x

# mypy app
black app --check
isort --recursive --check-only app
flake8 --exclude .git,__pycache__,.env,.mypy_cache --per-file-ignores="__init__.py:F401"