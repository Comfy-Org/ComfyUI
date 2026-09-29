set "BASE=%~dp0"

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="

"%BASE%python_embeded\python.exe" -s "%BASE%ComfyUI\main.py" --windows-standalone-build --fast
pause
