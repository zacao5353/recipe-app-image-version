#!/usr/bin/env bash
# render-build.sh

# OpenCVが依存する可能性のあるシステムライブラリをインストール
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Pythonの依存関係をインストール
pip install -r requirements.txt