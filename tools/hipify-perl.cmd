@echo off
setlocal
set "PERL_EXE=C:\Program Files\Git\usr\bin\perl.exe"
set "HIPIFY_SCRIPT=%~dp0..\sandbox\hipify\bin\hipify-perl"
if not exist "%PERL_EXE%" (
  echo Git for Windows Perl was not found at "%PERL_EXE%". 1>&2
  exit /b 1
)
if not exist "%HIPIFY_SCRIPT%" (
  echo Pinned HIPIFY checkout is missing; see scripts\sync-upstreams.ps1. 1>&2
  exit /b 1
)
"%PERL_EXE%" "%HIPIFY_SCRIPT%" %*
exit /b %errorlevel%
