# Native Windows setup. WSL2 is recommended, but this can be used if you want to try native Windows.
# Run from PowerShell inside the repo folder.

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Write-Host "Setup complete. Run: .\venv\Scripts\Activate.ps1 ; python app.py"
