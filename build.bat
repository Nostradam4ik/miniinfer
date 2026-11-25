@echo off
rem Build MiniInfer: run build.bat from any terminal.
rem Uses MSVC (Build Tools), Ninja and local CUDA from C:\Code\cuda-13.5.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set "CUDA_PATH=C:\Code\cuda-13.5"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" || exit /b 1
cmake --build "%~dp0build" || exit /b 1
