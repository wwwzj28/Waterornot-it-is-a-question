@echo off
chcp 65001 > nul
@rem Windows 一键运行入口
@rem 使用方法：把本文件放在 waterornot 根目录，双击即可运行完整流程。

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% scripts\12_run_all.py --mode all
pause
