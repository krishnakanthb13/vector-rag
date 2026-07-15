@echo off
title Vector RAG Toolkit
color 0B
cd /d "%~dp0"

:menu
cls
echo ============================================
echo    Vector RAG Toolkit
echo ============================================
echo.
echo   1. Build embeddings (first time or rebuild)
echo   2. Search the vector store
echo   3. Chat with your documents (interactive)
echo   4. View chat history
echo   5. Exit
echo.
set /p choice="  Select option [1-5]: "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto search
if "%choice%"=="3" goto chat
if "%choice%"=="4" goto history
if "%choice%"=="5" goto exit

echo Invalid option.
pause
goto menu

:build
cls
echo Building embeddings from source_docs/ ...
echo.
python scripts\build_embeddings.py
echo.
pause
goto menu

:search
cls
set /p query="  Enter search query: "
if "%query%"=="" goto menu
python scripts\query_embeddings.py "%query%"
echo.
pause
goto menu

:chat
cls
python scripts\chat.py
pause
goto menu

:history
cls
echo View chat history
echo.
echo   1. Show all
echo   2. Show last 10
echo   3. Search history
echo   4. Clear history
echo   5. Back
echo.
set /p hchoice="  Select [1-5]: "
if "%hchoice%"=="1" python scripts\view_history.py
if "%hchoice%"=="2" python scripts\view_history.py --last 10
if "%hchoice%"=="3" goto history_search
if "%hchoice%"=="4" python scripts\view_history.py --clear
if "%hchoice%"=="5" goto menu
echo.
pause
goto history

:history_search
set /p hquery="  Search text: "
if "%hquery%"=="" goto history
python scripts\view_history.py --search "%hquery%"
echo.
pause
goto history

:exit
exit
