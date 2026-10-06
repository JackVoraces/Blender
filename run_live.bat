@echo off
rem Avatar and tracker together.  Press Q in the tracker window to stop tracking.
call "%~dp0run_avatar.bat"
call "%~dp0run_tracker.bat" %*
