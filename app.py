"""
app.py — Flask API for Intelligent PDF Extraction & Highlighting

Endpoints:
  POST /upload          → Upload PDF, extract fields, get highlighted PDF
  GET  /download/<file> → Download highlighted PDF
  GET  /health          → Health check

Usage:
  python app.py

Then test:
  curl -X POST http://localhost:5000/upload \
       -F "file=@invoice.pdf" \
       -F "doc_type=invoice"
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from services.pdf_reader         import is_text_pdf, extract_text_and_words
from services.llm_service        import extract_fields_with_llm
from services.coordinate_matcher import match_all_fields
from services.highlighter        import highlight_pdf

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
ALLOWED_EXT   = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "PDF Extractor"}), 200


# ─── MAIN UPLOAD + PROCESS ENDPOINT ───────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_and_process():
    """
    Upload a PDF and get:
    - extracted JSON fields
    - downloadable highlighted PDF

    Form fields:
      file     : PDF file (required)
      doc_type : "invoice" | "kyc" | "resume" | "bank_statement" | "auto" (optional, default: auto)
    """

    # 1. Validate file
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send as multipart/form-data with key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    doc_type = request.form.get("doc_type", "auto")

    # 2. Save uploaded file
    file_id      = str(uuid.uuid4())[:8]
    original_name = os.path.splitext(file.filename)[0]
    upload_path  = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")
    file.save(upload_path)

    try:
        # 3. Detect PDF type
        text_based = is_text_pdf(upload_path)

        if not text_based:
            # OCR path — placeholder for PaddleOCR integration
            return jsonify({
                "error": "Scanned PDF detected. OCR support (PaddleOCR) coming soon.",
                "hint" : "Currently supports text-based PDFs only."
            }), 422
            
        # 4. Extract text + word coordinates
        extraction = extract_text_and_words(upload_path)
        full_text  = extraction["full_text"]
        words      = extraction["words"]

        print("full_text: ",full_text)
        if not full_text.strip():
            return jsonify({"error": "Could not extract any text from PDF"}), 422

        # 5. Send to Claude on Bedrock
        llm_result = extract_fields_with_llm(full_text, doc_type)
        fields     = llm_result["fields"]
        usage      = llm_result["usage"]

        if not fields:
            return jsonify({
                "error"  : "LLM returned no fields",
                "details": llm_result.get("error", "Unknown")
            }), 500

        # 6. Match fields to coordinates
        field_coords = match_all_fields(fields, words)

        # 7. Highlight PDF
        output_filename = f"highlighted_{file_id}_{original_name}.pdf"
        output_path     = os.path.join(OUTPUT_FOLDER, output_filename)
        highlight_pdf(upload_path, output_path, field_coords, add_label=True)

        # 8. Build response
        download_url = f"/download/{output_filename}"

        # Summarize which fields were found on-page
        highlight_summary = {
            field: len(coords) > 0
            for field, coords in field_coords.items()
        }

        return jsonify({
            "status"          : "success",
            "doc_type"        : doc_type,
            "extracted_fields": fields,
            "highlights_found": highlight_summary,
            "highlighted_pdf" : download_url,
            "token_usage"     : usage,
            "page_count"      : extraction["page_count"]
        }), 200

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# ─── DOWNLOAD ENDPOINT ────────────────────────────────────────────────────────

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename: str):
    """Download the highlighted PDF."""
    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        as_attachment=True,
        mimetype="application/pdf"
    )


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PDF Extraction & Highlighting Service")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)