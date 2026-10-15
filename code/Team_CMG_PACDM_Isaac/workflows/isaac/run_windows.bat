@echo off
rem Run with the Python of the active Isaac Sim environment.  All arguments are passed on.
setlocal
cd /d "%~dp0\..\.."
python workflows\isaac\run_isaac.py %*
exit /b %ERRORLEVEL%
