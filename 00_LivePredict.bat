@echo off
Title Python script execution
color 8F
mode con lines=10 cols=80
echo.
echo ...executing live predict script...
echo.

set "dir=C:\Users\Eichleitner\Documents\Coding"

echo.
echo 07_live_predict_v04.py wird gestartet!
python "%dir%\07_live_predict_v04.py" "%dir%"
timeout /t 2

echo.
echo ...terminating script...
echo.
echo.
timeout /t 3 