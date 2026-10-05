@echo off
REM Chay AI Xuong Co Khi tren Windows. Lan dau se tu cai thu vien.
cd /d "%~dp0"
if not exist ".venv" (
  echo Tao moi truong ao .venv ...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  echo Cai thu vien - lan dau mat vai phut vi TensorFlow kha nang ...
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python -c "import dotenv" >nul 2>&1
if errorlevel 1 (
  echo Cap nhat thu vien con thieu ...
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
  echo [CANH BAO] TensorFlow khong khoi dong duoc; AI se chay o che do trai nghiem.
  .venv\Scripts\python -c "import sys; print('Python goc:', sys.base_prefix)"
  echo Neu Python goc nam trong Miniconda/Anaconda, xem muc sua loi Windows trong README.md.
)
.venv\Scripts\python -m streamlit run app.py %*
