@echo off
rem Build MiniInfer from any Windows terminal. Extra arguments go to the CMake configure step, e.g.
rem   build.bat -DCMAKE_CUDA_ARCHITECTURES=86
rem Needs Visual Studio 2022 (or Build Tools) with the C++ and CMake components, and the CUDA Toolkit
rem with CUDA_PATH set.

if not defined CUDA_PATH (
    echo error: CUDA_PATH is not set. Point it to your CUDA Toolkit directory.
    exit /b 1
)

if not defined VSCMD_VER (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    call :find_vs || exit /b 1
)

set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
pushd "%~dp0"
cmake --preset release -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" %* || goto :fail
cmake --build --preset release || goto :fail
popd
exit /b 0

:fail
popd
exit /b 1

:find_vs
if not exist "%VSWHERE%" (
    echo error: vswhere.exe not found. Install Visual Studio 2022 or run from a Developer Command Prompt.
    exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo error: no Visual Studio installation with the C++ toolset was found.
    exit /b 1
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
exit /b 0
