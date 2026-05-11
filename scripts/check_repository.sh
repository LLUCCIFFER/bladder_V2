#!/usr/bin/env bash
set -euo pipefail

python -m py_compile ./*.py
git status --short --ignored

