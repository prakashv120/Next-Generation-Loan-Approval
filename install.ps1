# Installation Script for Gen-Z Credit Scoring Engine
# This script installs all required dependencies

Write-Host ""
Write-Host "=" * 60
Write-Host "Installing Gen-Z Credit Scoring Engine Dependencies" -ForegroundColor Cyan
Write-Host "=" * 60
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.7 or higher." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
Write-Host ""

# Install packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "=" * 60
Write-Host "✓ Installation Complete!" -ForegroundColor Green
Write-Host "=" * 60
Write-Host ""
Write-Host "To run the application, simply type:" -ForegroundColor Cyan
Write-Host "    python run.py" -ForegroundColor Yellow
Write-Host ""
