"""
MarkItDown Converter — FastAPI backend.

Usage:
    cd converter
    uvicorn app:app --reload --port 8000
    # open http://localhost:8000
"""

import os
import pathlib
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markitdown import MarkItDown

app = FastAPI(title="MarkItDown Converter")

# Allow requests from VS Code Live Server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_md = MarkItDown()

_static = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".txt", ".md", ".csv", ".json", ".xml",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".zip",
}
MAX_SIZE_MB = 50


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((_static / "index.html").read_text(encoding="utf-8"))


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """
    Accept a file upload, run markitdown conversion, return JSON:
        { "filename": str, "markdown": str, "token_estimate": int }
    Token estimate uses the ~4 chars/token heuristic (works for Claude & GPT).
    """
    suffix = pathlib.Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_SIZE_MB} MB limit.",
        )

    # Write to a temp file so markitdown can open it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _md.convert(tmp_path)
        markdown = result.text_content or ""
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Conversion failed: {exc}"
        ) from exc
    finally:
        os.unlink(tmp_path)

    return JSONResponse(
        {
            "filename": file.filename,
            "markdown": markdown,
            "token_estimate": max(1, len(markdown) // 4),
        }
    )
