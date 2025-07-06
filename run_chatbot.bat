@echo off
echo ===================================
echo AI Chatbot Application Launcher
echo ===================================
echo.

echo Activating virtual environment...
IF NOT EXIST venv\Scripts\activate.bat (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: pip install -r requirements.txt
    echo Then run: python nltk_download.py
    pause
    exit /b 1
)

CALL venv\Scripts\activate.bat

echo.
echo Starting AI Chatbot Application...
echo.
echo Access the application at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python main.py

echo.
echo Application has stopped.
pause
