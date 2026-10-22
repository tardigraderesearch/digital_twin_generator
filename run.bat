@echo off
rem Digital Twin Generator launcher for Windows. Double-click it, or run:
rem   run.bat                      open the app
rem   run.bat health SCAN.obj      any dtg command works too
rem
rem This file only finds Python 3.11+ and hands over to launcher.py, which does
rem the rest on every OS: a private .venv next to this file (first run only),
rem installing the dependencies when they change, and launching. Nothing is
rem installed system-wide.
rem Set DTG_PYTHON=C:\path\to\python.exe to pick a Python.
rem
rem Kept deliberately simple: no parenthesised blocks or for /f, because cmd
rem mis-parses Python code inside them.

setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "PY="
if defined DTG_PYTHON call :try "%DTG_PYTHON%"
call :try py -3.13
call :try py -3.12
call :try py -3.11
call :try py -3
call :try python
call :try python3
if defined PY goto :found

echo.
echo [digital-twin-generator] PROBLEM: Python 3.11 or newer was not found.
echo     Install it from https://www.python.org/downloads/windows/
echo     In the installer, tick "Add python.exe to PATH" and keep "tcl/tk and IDLE" ticked.
echo     Then double-click run.bat again.
echo.
pause
exit /b 1

:found
%PY% "%~dp0launcher.py" %*
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" exit /b 0
if "%~1"=="" pause
exit /b %CODE%

:try
if defined PY goto :eof
%* -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if errorlevel 1 goto :eof
set PY=%*
goto :eof
