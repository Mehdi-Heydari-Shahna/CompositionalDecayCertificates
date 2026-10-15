#!/usr/bin/env bash
# The verification campaign with the Python of the active Isaac Sim environment.  All arguments are passed on:
#   workflows/isaac/run_campaign_linux.sh --headless --quick
#   workflows/isaac/run_campaign_linux.sh --headless --resume results/isaac_campaign_ID
set -e
cd "$(dirname "$0")/../.."
exec python workflows/isaac/run_campaign.py "$@"
