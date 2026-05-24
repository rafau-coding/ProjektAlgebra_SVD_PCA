@echo off
chcp 65001 >nul
echo Zmieniam rozszerzenia plików na .jpg w folderze "moje_zdjecia"...

:: Przejście do folderu moje_zdjecia
cd "moje_zdjecia"

:: Sprawdzenie, czy folder istnieje
if %errorlevel% neq 0 (
    echo.
    echo BŁĄD: Nie znaleziono folderu "moje_zdjecia"! 
    echo Upewnij się, że ten skrypt znajduje się w tym samym miejscu co folder.
    echo.
    pause
    exit /b
)

:: Zmiana wszystkich rozszerzeń na .jpg
ren *.* *.jpg

echo.
echo Gotowe! Wszystkie pliki w folderze mają teraz rozszerzenie .jpg.
pause
