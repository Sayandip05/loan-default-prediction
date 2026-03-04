@echo off
REM ========================================
REM Start All Services
REM ========================================

echo.
echo ========================================
echo  Starting Loan Default Prediction App
echo ========================================
echo.

call venv\Scripts\activate.bat

echo.
echo 🚀 Starting services...
echo.
echo Services will be available at:
echo   - Streamlit UI:  http://localhost:8501
echo   - FastAPI:       http://localhost:8000/docs
echo   - MLflow UI:     http://localhost:5000
echo.

REM Start MLflow in background
start "MLflow" cmd /c "venv\Scripts\activate.bat && mlflow ui --host 0.0.0.0 --port 5000"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start FastAPI in background
start "FastAPI" cmd /c "venv\Scripts\activate.bat && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start Streamlit (main window)
echo.
echo ✅ All services starting...
echo.
echo Press Ctrl+C in each window to stop services
echo.
streamlit run frontend\streamlit_app.py

pause
