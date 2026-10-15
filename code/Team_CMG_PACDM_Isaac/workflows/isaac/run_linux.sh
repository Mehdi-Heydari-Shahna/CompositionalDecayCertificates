#!/usr/bin/env bash
# Run with the Python of the active Isaac Sim environment.  All arguments are passed on.
set -e
cd "$(dirname "$0")/../.."
exec python workflows/isaac/run_isaac.py "$@"
