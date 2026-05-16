@echo off
chcp 65001 > nul
@rem Windows 一键入口：生成最终交付包
@rem 使用方法：把本文件放在项目根目录，双击即可重建 final/ 输出。

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% scripts\17_eval_final_hybrid.py
pause
