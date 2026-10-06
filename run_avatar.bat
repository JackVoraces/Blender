@echo off
rem The Godot avatar window.  Listens for the tracker on UDP 127.0.0.1:8765.
cd /d "%~dp0"
set "GODOT="
for %%G in ("tools\godot\Godot*win64.exe") do set "GODOT=%%G"
if not defined GODOT (
    echo No Godot binary found in tools\godot. See README.md, "Setting up on another machine".
    pause
    exit /b 1
)
start "" "%GODOT%" --path "%~dp0godot" %*
