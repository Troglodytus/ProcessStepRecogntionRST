@echo off
Title Python script execution
color 8F
mode con lines=10 cols=80
echo.
echo Commencing AI preprocessing scripts
echo.

set "dir=C:\Users\Eichleitner\Documents\Coding"

echo.
echo 02_resize_images.py wird gestartet!
python "%dir%\02_resize_images.py" "%dir%"
timeout /t 2

echo.
echo 03a_convert_format.py wird gestartet!
python "%dir%\03a_convert_format.py" "%dir%"
timeout /t 2

echo.
echo 03a_rename_category.py wird gestartet!
python "%dir%\03a_rename_category.py" "%dir%"
timeout /t 2

echo.
echo 04a_label_csv_combo.py wird gestartet!
python "%dir%\04a_label_csv_combo.py" "%dir%"
timeout /t 2

echo.
echo 05_train_cnn_v18.py wird gestartet!
python "%dir%\05_train_cnn_v18.py" "%dir%" - 13 Classes
timeout /t 2

echo.
echo All script have been executed
echo.
echo.
timeout /t 3 