@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please complete the Python 3.12 setup in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m vjms %*
