@echo off
rem Webcam face tracker with its tuning window.  Extra arguments are passed through:
rem   run_tracker.bat --list-cameras
rem   run_tracker.bat --source synthetic        (scripted face, no camera)
rem   run_tracker.bat --source image            (sample portrait through the real tracker)
cd /d "%~dp0"
set GLOG_minloglevel=2
".venv\Scripts\python.exe" -m tracker %*
if errorlevel 1 pause
