#!/usr/bin/env bash
# Chạy AI Xưởng Cơ Khí trên macOS / Linux. Lần đầu sẽ tự cài thư viện.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d ".venv" ]; then
  echo "▶ Tạo môi trường ảo .venv ..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  echo "▶ Cài thư viện (lần đầu mất vài phút vì TensorFlow khá nặng) ..."
  .venv/bin/python -m pip install -r requirements.txt
fi
if ! .venv/bin/python -c 'import dotenv' >/dev/null 2>&1; then
  echo "▶ Cài các thư viện mới còn thiếu ..."
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m streamlit run app.py "$@"
