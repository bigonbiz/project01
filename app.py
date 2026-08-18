from pathlib import Path
import io
import os
import socket
import tempfile

import fitz  # PyMuPDF
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
# Paddle's Windows C++ predictor cannot open model files through this
# workspace's Korean path, so keep the model cache at an ASCII local path.
MODEL_CACHE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "paddleocr_local_models"
UPLOAD_LIMIT = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "pdf"}

# Keep Paddle/PaddleX caches writable and local to this workspace. This must be
# set before importing PaddleOCR because Paddle initializes its cache on import.
os.environ.setdefault("PADDLE_HOME", str(MODEL_CACHE / "paddle"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE))
os.environ.setdefault("PADDLE_OCR_BASE_DIR", str(MODEL_CACHE / "paddleocr"))
os.environ.setdefault("FLAGS_use_mkldnn", "False")
os.environ.setdefault("FLAGS_use_onednn", "False")

from paddleocr import PaddleOCR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>로컬 PaddleOCR</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#17202a;background:#f6f8fb}
main{background:white;padding:28px;border-radius:14px;box-shadow:0 4px 20px #0001}h1{margin-top:0}
input[type=file]{margin:12px 0}button{background:#1769e0;color:#fff;border:0;border-radius:8px;padding:11px 18px;font-size:15px;cursor:pointer}
.hint{color:#667085;font-size:14px}.error{background:#fff0f0;border:1px solid #f5b5b5;padding:12px;border-radius:8px}
.result{margin-top:22px;border-top:1px solid #e5e7eb;padding-top:16px}.file{font-weight:700;margin-bottom:6px}.text{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;padding:14px;border-radius:8px;min-height:32px}
</style></head><body><main>
<h1>로컬 PaddleOCR</h1><p class="hint">이미지 또는 PDF를 최대 10개까지 선택하세요. 파일은 외부 OCR API로 전송되지 않습니다.</p>
<form method="post" enctype="multipart/form-data"><input type="file" name="files" accept="image/*,.pdf" multiple required><br><button type="submit">OCR 실행</button></form>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% for item in results %}<section class="result"><div class="file">{{ item.name }}</div>{% for page in item.pages %}<div class="hint">{{ page.label }}</div><div class="text">{{ page.text }}</div>{% endfor %}</section>{% endfor %}
</main></body></html>"""


def make_ocr():
    # Korean OCR model; all inference happens in this process on local files.
    return PaddleOCR(
        lang="korean",
        use_angle_cls=False,
        enable_mkldnn=False,
        show_log=False,
    )


OCR = None


def ocr_image(ocr, image_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    result = ocr.ocr(image, cls=False)
    lines = []
    for page in result or []:
        for item in page or []:
            text, score = item[1]
            lines.append(f"{text}  [{score:.3f}]")
    return "\n".join(lines) or "(인식된 텍스트가 없습니다.)"


def pages_for_file(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix != ".pdf":
        return [("이미지", raw)]
    doc = fitz.open(stream=raw, filetype="pdf")
    pages = []
    for page_no, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pages.append((f"페이지 {page_no}", pix.tobytes("png")))
    doc.close()
    return pages


@app.route("/", methods=["GET", "POST"])
def index():
    global OCR
    results, error = [], None
    if request.method == "POST":
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            error = "파일을 선택해 주세요."
        elif len(files) > UPLOAD_LIMIT:
            error = "한 번에 최대 10개 파일까지 업로드할 수 있습니다."
        else:
            invalid = [f.filename for f in files if Path(f.filename).suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS]
            if invalid:
                error = "지원하지 않는 파일 형식: " + ", ".join(invalid)
            else:
                if OCR is None:
                    OCR = make_ocr()
                for file in files:
                    raw = file.read()
                    pages = [{"label": label, "text": ocr_image(OCR, image)} for label, image in pages_for_file(file.filename, raw)]
                    results.append({"name": file.filename, "pages": pages})
    return render_template_string(HTML, results=results, error=error)


def free_port(start=8765):
    for port in range(start, start + 100):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("사용 가능한 localhost 포트를 찾지 못했습니다.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", free_port()))
    print(f"Local OCR app: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
