#!/usr/bin/env bash
set -euo pipefail
# Run this script on your LOCAL machine, not on Snellius.
# Edit LOCAL_LOBSTER_DIR to point at the folder containing *_message_10.csv and *_orderbook_10.csv.

LOCAL_LOBSTER_DIR="${LOCAL_LOBSTER_DIR:-/local/path/to/AMZN/2024}"
SNELLIUS_USER="${SNELLIUS_USER:-adong}"
SNELLIUS_HOST="${SNELLIUS_HOST:-snellius.surf.nl}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/home/adong/JaxMARL-HFT/data/rawLOBSTER/AMZN/2024}"

rsync -avP "${LOCAL_LOBSTER_DIR}/" \
  "${SNELLIUS_USER}@${SNELLIUS_HOST}:${REMOTE_DATA_DIR}/"
