@echo off
rem The verification campaign with the Python of the active Isaac Sim environment.  All arguments are passed on:
rem   workflows\isaac\run_campaign_windows.bat --headless --quick
rem   workflows\isaac\run_campaign_windows.bat --headless --resume results\isaac_campaign_ID
setlocal
cd /d "%~dp0\..\.."
python workflows\isaac\run_campaign.py %*
exit /b %ERRORLEVEL%
