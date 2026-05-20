@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Softmax -- Nsight Compute Profiles
echo ========================================
echo.

call conda activate ainfra
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env 'ainfra'
    pause
    exit /b 1
)

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,4%
set PROF_DIR=profiles\%TIMESTAMP%
if not exist "%PROF_DIR%\" mkdir "%PROF_DIR%"

echo Output: %PROF_DIR%\
echo.

set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\ncu.bat"
set PYTHON=python
set SCRIPT=bench/profile_kernel.py

echo [1/5] Profiling Naive Softmax...
call %NCU% --set full -k regex:softmax_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_01_naive %PYTHON% %SCRIPT% softmax_kernel
echo.

echo [2/5] Profiling Online Softmax...
call %NCU% --set full -k regex:online_softmax_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_02_online %PYTHON% %SCRIPT% online_softmax_kernel
echo.

echo [3/5] Profiling Warp Softmax...
call %NCU% --set full -k regex:softmax_warp_online_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_03_warp %PYTHON% %SCRIPT% softmax_warp_online_kernel
echo.

echo [4/5] Profiling Warp+float4 Softmax...
call %NCU% --set full -k regex:softmax_warp_float4_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_04_warp_float4 %PYTHON% %SCRIPT% softmax_warp_float4_kernel
echo.

echo [5/5] Profiling Warp+Tiled Softmax...
call %NCU% --set full -k regex:softmax_warp_tiled_kernel$ -s 5 -c 1 -o %PROF_DIR%\profile_05_warp_tiled %PYTHON% %SCRIPT% softmax_warp_tiled_kernel
echo.

echo ========================================
echo   All profiles complete.
echo   Open with: ncu-ui %PROF_DIR%\*.ncu-rep
echo ========================================
pause
