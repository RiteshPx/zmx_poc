"""
services/pdf_reader.py

Hybrid PDF reader:
  - PyMuPDF  → word-level bounding box COORDINATES (precise, for text PDFs)
  - PaddleOCR → text EXTRACTION (accurate OCR, handles scanned/image PDFs)

Output format is IDENTICAL to the previous version —
no changes needed in app.py, llm_service.py, coordinate_matcher.py, or highlighter.py.

Install:
    pip install pymupdf paddlepaddle paddleocr
"""

import os
import fitz                          # PyMuPDF  — coordinates
import numpy as np
from typing import Any

# Work around Paddle oneDNN/PIR runtime incompatibilities seen in some envs.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

try:
    from paddleocr import PaddleOCR  # PaddleOCR — text extraction
    _paddleocr_import_error = None
except Exception:  # pragma: no cover - handled at runtime with clear error
    PaddleOCR = None
    _paddleocr_import_error = "Unknown import error"
    try:
        import traceback
        _paddleocr_import_error = traceback.format_exc().strip()
    except Exception:
        pass


# ─── PaddleOCR singleton (heavy to init, load once) ───────────────────────────
_ocr_engine = None

def _get_ocr() -> Any:
    global _ocr_engine
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR is not available. Install dependencies with:\n"
            "  pip install paddlepaddle paddleocr\n"
            f"Import error details:\n{_paddleocr_import_error}"
        )
    if _ocr_engine is None:
        try:
            _ocr_engine = PaddleOCR(
                use_angle_cls=True,   # handles rotated text
                lang="en",            # change to "ch", "hi", etc. if needed
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "paddle_static" in msg and "paddlepaddle" in msg:
                raise RuntimeError(
                    "PaddleOCR engine is unavailable because 'paddlepaddle' "
                    "is missing. Install with:\n"
                    "  pip install paddlepaddle paddleocr\n"
                    "If using Google Colab, restart runtime after install."
                ) from exc
            raise
        except AttributeError as exc:
            msg = str(exc)
            if "AnalysisConfig" in msg and "set_optimization_level" in msg:
                raise RuntimeError(
                    "PaddleOCR and PaddlePaddle versions are incompatible in "
                    "this environment. Install pinned compatible versions:\n"
                    "  pip uninstall -y paddleocr paddlepaddle\n"
                    "  pip install paddlepaddle==2.6.2 paddleocr==2.7.3\n"
                    "Then restart runtime/server and retry."
                ) from exc
            raise
    return _ocr_engine


# ─── PUBLIC: PDF type detection (unchanged interface) ─────────────────────────

def is_text_pdf(pdf_path: str) -> bool:
    """
    Returns True  → text-based PDF (has selectable text via PyMuPDF).
    Returns False → scanned/image PDF (needs full OCR path).

    Used by app.py to decide routing — interface unchanged.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text().strip():
            doc.close()
            return True
    doc.close()
    return False


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _page_to_numpy(page: fitz.Page, dpi: int = 200):
    """
    Render a PyMuPDF page to a numpy array for PaddleOCR.
    Higher DPI = better OCR accuracy on scanned docs.
    """
    mat    = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img    = np.frombuffer(pixmap.samples, dtype=np.uint8)
    img    = img.reshape(pixmap.height, pixmap.width, 3)
    return img, pixmap.width, pixmap.height


def _scale_coords(x0, y0, x1, y1, img_w, img_h, page_w, page_h) -> tuple:
    """
    PaddleOCR returns pixel coordinates on the rendered image.
    Scale them back to PDF point coordinates (PyMuPDF space).
    """
    sx = page_w / img_w
    sy = page_h / img_h
    return (
        round(x0 * sx, 2),
        round(y0 * sy, 2),
        round(x1 * sx, 2),
        round(y1 * sy, 2)
    )


def _paddle_results_to_words(ocr_result, page_num: int,
                              img_w: int, img_h: int,
                              page_w: float, page_h: float) -> tuple:
    """
    Convert raw PaddleOCR output → (full_text str, word dicts list).

    PaddleOCR result structure per line:
        [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", confidence) ]
    """
    lines     = []
    word_list = []

    if not ocr_result or ocr_result[0] is None:
        return "", []

    for line in ocr_result[0]:
        box, (text, conf) = line
        if not text.strip():
            continue

        # Bounding quad → axis-aligned rect
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        px0, py0, px1, py1 = min(xs), min(ys), max(xs), max(ys)

        # Scale from image pixels → PDF points
        x0, y0, x1, y1 = _scale_coords(
            px0, py0, px1, py1,
            img_w, img_h, page_w, page_h
        )

        lines.append(text.strip())

        # Split multi-word lines into individual word entries
        # Each word gets an equal horizontal slice of the line box
        words = text.strip().split()
        n     = len(words)
        width = (x1 - x0) / n if n > 0 else (x1 - x0)

        for i, word in enumerate(words):
            word_list.append({
                "word": word,
                "page": page_num,
                "x0"  : round(x0 + i * width, 2),
                "y0"  : round(y0, 2),
                "x1"  : round(x0 + (i + 1) * width, 2),
                "y1"  : round(y1, 2)
            })

    return "\n".join(lines), word_list


def _run_ocr(ocr: Any, img_array: np.ndarray):
    """
    Run OCR with compatibility across PaddleOCR versions.
    Older versions accept `cls=True`, newer `predict()` path may not.
    """
    try:
        return ocr.ocr(img_array, cls=True)
    except TypeError as exc:
        if "unexpected keyword argument 'cls'" in str(exc):
            return ocr.ocr(img_array)
        raise
    except RuntimeError as exc:
        msg = str(exc)
        if "ConvertPirAttribute2RuntimeAttribute not support" in msg:
            raise RuntimeError(
                "Paddle runtime failed due to oneDNN/PIR compatibility issue. "
                "Please use compatible versions and restart runtime.\n"
                "Recommended: paddlepaddle>=2.6.0,<3.0.0 and paddleocr>=2.7.0."
            ) from exc
        raise


# ─── PUBLIC: Main extraction function (SAME interface as before) ───────────────

def extract_text_and_words(pdf_path: str, dpi: int = 200) -> dict:
    """
    Extract full text (via PaddleOCR) + per-word coordinates (via PyMuPDF).

    Strategy per page:
      Text PDF  → PyMuPDF gives precise coordinates.
                  PaddleOCR runs on rendered image for accurate text.
      Scanned   → PyMuPDF renders page to image.
                  PaddleOCR extracts text + approximate word boxes.
                  Coordinates are scaled back to PDF point space.

    Returns (IDENTICAL schema to previous version):
        {
            "full_text"  : "Invoice No: INV-101 ...",
            "words"      : [
                {
                    "word": "INV-101",
                    "page": 0,
                    "x0": 120.5, "y0": 200.1,
                    "x1": 180.3, "y1": 215.6
                },
                ...
            ],
            "page_count" : 2
        }
    """
    doc        = fitz.open(pdf_path)
    ocr        = _get_ocr()
    full_parts = []
    all_words  = []
    for page_num, page in enumerate(doc):
        page_w = page.rect.width
        page_h = page.rect.height

        # ── Render page to image (needed for PaddleOCR in both cases) ──────────
        img_array, img_w, img_h = _page_to_numpy(page, dpi=dpi)

        # ── PaddleOCR → TEXT (runs on rendered image, works for both types) ────
        ocr_result = _run_ocr(ocr, img_array)
        page_text, ocr_words = _paddle_results_to_words(
            ocr_result, page_num, img_w, img_h, page_w, page_h
        )
        if page.get_text().strip():
            # ── TEXT PDF: PyMuPDF → precise COORDINATES ───────────────────────
            pymupdf_words = page.get_text("words")
            # items: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            for w in pymupdf_words:
                x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
                all_words.append({
                    "word": word_text.strip(),
                    "page": page_num,
                    "x0"  : round(x0, 2),
                    "y0"  : round(y0, 2),
                    "x1"  : round(x1, 2),
                    "y1"  : round(y1, 2)
                })
        else:
            # ── SCANNED PDF: PaddleOCR → scaled COORDINATES ───────────────────
            all_words.extend(ocr_words)

        full_parts.append(page_text)

    doc.close()

    return {
        "full_text" : "\n".join(full_parts),
        "words"     : all_words,
        "page_count": len(full_parts)
    }