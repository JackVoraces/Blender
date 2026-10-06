@echo off
rem Python tests (rig definition, tracker chain, real MediaPipe checks), then the
rem Blender/Godot parity test.
cd /d "%~dp0"
set GLOG_minloglevel=2
".venv\Scripts\python.exe" -m unittest discover -s tests -t .
for %%G in ("tools\godot\Godot*_console.exe") do "%%G" --headless --path "%~dp0godot" -s res://tests/parity_test.gd
pause
