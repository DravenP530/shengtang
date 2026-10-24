@echo off
cd /d "%~dp0"
set PYTHONPATH=
set PYTHONNOUSERSITE=1
set DSH_HOME=
set "PY="
if exist "%USERPROFILE%\.workbuddy-ai\binaries\python\envs\dsh-sdk\Scripts\python.exe" (
  set "PY=%USERPROFILE%\.workbuddy-ai\binaries\python\envs\dsh-sdk\Scripts\python.exe"
)
if not defined PY (
  where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python was not found. Install Python 3, then double-click again.
  pause
  exit /b 1
)
echo.
echo   Looking for a local model, then opening http://127.0.0.1:8770
echo   Close this window to stop.
echo.
%PY% -m stage.launch
echo.
echo   Stopped.
pause
