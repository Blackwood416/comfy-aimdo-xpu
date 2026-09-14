@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build\windows-reader-unit"
if not defined VS_PATH (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS_PATH=%%I"
)
call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
if not defined WINDOWS_SDK_NUGET set "WINDOWS_SDK_NUGET=%ROOT_DIR%\build\windows-sdk-nuget"
if not defined WINDOWS_SDK_VERSION set "WINDOWS_SDK_VERSION=10.0.26100.0"
if exist "%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt\stddef.h" (
    set "INCLUDE=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\shared;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\um;%INCLUDE%"
    set "LIB=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\ucrt\x64;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\um\x64;%LIB%"
)
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cl.exe /nologo /O2 /MD /W3 /DAIMDO_XPU /I"%ROOT_DIR%\src" ^
    "%ROOT_DIR%\tests\windows_reader_unit.c" ^
    /Fo"%BUILD_DIR%\\" /Fe"%BUILD_DIR%\windows_reader_unit.exe"
if errorlevel 1 exit /b 1
"%BUILD_DIR%\windows_reader_unit.exe"
if errorlevel 1 exit /b 1
endlocal
