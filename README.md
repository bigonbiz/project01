# Local PaddleOCR Web App

Local Flask web app for OCR of up to 10 image/PDF files. OCR inference runs locally with PaddleOCR; uploaded files are not sent to an external OCR API.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the localhost URL printed by the app. The app selects an available port beginning at 8765.
