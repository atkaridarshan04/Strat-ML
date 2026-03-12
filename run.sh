#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
PYTHONPATH=$(pwd) python3 cli/main.py "$@"
