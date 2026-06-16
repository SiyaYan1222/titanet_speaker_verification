#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3.10 python3.10-venv ffmpeg libsndfile1 git git-lfs

git lfs install || true

python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup complete. Run: source venv/bin/activate && python app.py"
