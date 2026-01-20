@echo off
echo ============================================================================
echo Computing Confidence Intervals for Fairness Benchmarks
echo ============================================================================
echo.

cd /d "%~dp0"

python scripts/add_confidence_intervals_to_benchmarks.py

echo.
echo ============================================================================
pause
