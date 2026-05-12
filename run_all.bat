@echo off
chcp 65001 > nul
REM Windows 一键运行入口
REM 使用方法：把本文件放在 waterornot 根目录，双击即可运行完整流程。

python scripts\12_run_all.py --mode all
pause
