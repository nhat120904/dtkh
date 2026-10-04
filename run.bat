@echo off
REM Chay AI Xuong Co Khi tren Windows. Lan dau se tu cai thu vien.
cd /d "%~dp0"
if not exist ".venv" (
  echo Tao moi truong ao .venv ...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  echo Cai thu vien - lan dau mat vai phut vi TensorFlow kha nang ...
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -m streamlit run app.py %*
