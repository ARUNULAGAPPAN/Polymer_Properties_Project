@echo off
cd /d "c:\Users\Arun\Desktop\PolySights"
echo ====================================================
echo  PolySights - Polymer Property Predictor
echo ====================================================
echo.
echo Checking for app.py...
if not exist app.py (
    if exist app_new.py (
        echo Updating app files...
        del app_old.py 2>nul
        ren app.py app_old.py 2>nul
        ren app_new.py app.py
        echo Done!
    )
)
echo.
echo Starting Streamlit app...
echo Open your browser to: http://localhost:8501
echo.
python -m streamlit run app.py
pause
