@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
echo 1. Tai du lieu GHAP (fetch)
echo 2. Phan tich du lieu da tai (analyze)
echo 3. Thoat
choice /c 123 /n /m "Chon 1, 2 hoac 3: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto analyze
py -3.13 "src\ghap_spatial_check.py" fetch --project project-0065a06f-7ddd-405b-82c
goto done
:analyze
py -3.13 "src\ghap_spatial_check.py" analyze
:done
if errorlevel 1 echo Chay chua thanh cong. Xem thong bao loi o tren.
echo.
pause
endlocal
