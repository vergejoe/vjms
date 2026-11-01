#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Please complete the Python 3.12 setup in README.md first."
  exit 1
fi
exec .venv/bin/python -m vjms "$@"
