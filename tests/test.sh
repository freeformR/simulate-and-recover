#!/bin/bash
echo "Running tests..."
PYTHONPATH="${PYTHONPATH}:$(pwd)/src" python3 -m unittest discover -s tests -v