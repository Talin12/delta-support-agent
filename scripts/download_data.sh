#!/usr/bin/env bash
# Download the raw Kaggle dataset. Requires ~/.kaggle/kaggle.json
# (kaggle.com -> Settings -> API -> Create New Token).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
mkdir -p "$RAW_DIR"

if [ -f "$RAW_DIR/twcs.csv" ]; then
  echo "twcs.csv already present, skipping download."
  exit 0
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found."
  echo "Create a token at https://www.kaggle.com/settings (API -> Create New Token), then:"
  echo "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

"$REPO_ROOT/.venv/bin/kaggle" datasets download \
  -d thoughtvector/customer-support-on-twitter -p "$RAW_DIR"

unzip -o "$RAW_DIR/customer-support-on-twitter.zip" -d "$RAW_DIR"
# The archive nests the csv under sample/ in some versions.
if [ ! -f "$RAW_DIR/twcs.csv" ]; then
  found="$(find "$RAW_DIR" -name 'twcs.csv' | head -1)"
  [ -n "$found" ] && mv "$found" "$RAW_DIR/twcs.csv"
fi
rm -f "$RAW_DIR/customer-support-on-twitter.zip"
ls -lh "$RAW_DIR/twcs.csv"
