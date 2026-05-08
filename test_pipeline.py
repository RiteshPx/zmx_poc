"""
test_pipeline.py

Run the full extraction pipeline locally WITHOUT starting Flask.
Creates a sample invoice PDF, processes it end-to-end, and saves the highlighted output.

Usage:
    python test_pipeline.py
"""

import os
import json
import fitz   # PyMuPDF — also used to generate the sample PDF
from dotenv import load_dotenv

load_dotenv()

from services.pdf_reader         import is_text_pdf, extract_text_and_words
from services.llm_service        import extract_fields_with_llm
from services.coordinate_matcher import match_all_fields
from services.highlighter        import highlight_pdf


# ─── 1. Create a Sample Invoice PDF ───────────────────────────────────────────

def create_sample_invoice(path: str):
    """Generate a simple invoice PDF for testing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    lines = [
        ("INVOICE",                    60, 760, 20, True),
        ("Invoice Number: INV-2026-001", 60, 720, 11, False),
        ("Invoice Date:   07-05-2026",  60, 703, 11, False),
        ("Due Date:       07-06-2026",  60, 686, 11, False),
        ("",                            60, 669, 11, False),
        ("Vendor:  ABC Technologies Pvt Ltd", 60, 652, 11, False),
        ("Address: 42 MG Road, Pune 411001", 60, 635, 11, False),
        ("",                            60, 618, 11, False),
        ("Bill To: Ritesh Parmar",      60, 601, 11, False),
        ("Company: XYZ Solutions",      60, 584, 11, False),
        ("",                            60, 567, 11, False),
        ("Description         Qty   Rate      Amount",   60, 540, 10, False),
        ("Cloud Services       1    40000    40000.00",   60, 523, 10, False),
        ("Support Package      1     5000     5000.00",   60, 506, 10, False),
        ("",                            60, 489, 10, False),
        ("Subtotal: Rs 45,000",         60, 465, 11, False),
        ("GST (18%): Rs 8,100",         60, 448, 11, False),
        ("Total Amount: Rs 53,100",     60, 425, 13, True),
        ("",                            60, 408, 11, False),
        ("Thank you for your business!", 60, 380, 10, False),
    ]

    for text, x, y, size, bold in lines:
        if not text:
            continue
        font = "helv" if not bold else "hebo"
        page.insert_text(
            fitz.Point(x, y),
            text,
            fontname=font,
            fontsize=size,
            color=(0, 0, 0)
        )

    doc.save(path)
    doc.close()
    print(f"✅ Sample PDF created: {path}")


# ─── 2. Run the Full Pipeline ─────────────────────────────────────────────────

def run_pipeline(pdf_path: str, doc_type: str = "invoice"):
    print("\n" + "=" * 65)
    print("  PDF Intelligence Pipeline — Local Test")
    print("=" * 65)

    # Step 1: Detect type
    text_based = is_text_pdf(pdf_path)
    print(f"\n[1] PDF Type        : {'Text-based ✅' if text_based else 'Scanned (OCR needed)'}")

    if not text_based:
        print("    Scanned PDFs require PaddleOCR — skipping for this test.")
        return

    # Step 2: Extract text + words
    extraction = extract_text_and_words(pdf_path)
    print(f"[2] Pages           : {extraction['page_count']}")
    print(f"    Words found     : {len(extraction['words'])}")
    print(f"    Text preview    : {extraction['full_text'][:120].strip()!r}...")

    # Step 3: LLM extraction
    print(f"\n[3] Sending to Claude on Bedrock...")
    llm_result = extract_fields_with_llm(extraction["full_text"], doc_type)
    fields     = llm_result["fields"]
    usage      = llm_result["usage"]

    print(f"    Input tokens    : {usage.get('input_tokens')}")
    print(f"    Output tokens   : {usage.get('output_tokens')}")
    print(f"\n    Extracted Fields:")
    print(json.dumps(fields, indent=6, ensure_ascii=False))

    # Step 4: Coordinate matching
    field_coords = match_all_fields(fields, extraction["words"])
    print(f"\n[4] Coordinate Matching:")
    for field, coords in field_coords.items():
        status = f"✅ found at {len(coords)} location(s)" if coords else "⚠️  not located in PDF"
        print(f"    {field:<22}: {status}")

    # Step 5: Highlight
    output_path = "outputs/highlighted_invoice.pdf"
    os.makedirs("outputs", exist_ok=True)
    highlight_pdf(pdf_path, output_path, field_coords, add_label=True)
    print(f"\n[5] Highlighted PDF : {output_path}")

    print("\n🎉 Pipeline complete!")
    print("=" * 65)
    return fields, output_path


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_path = "uploads/sample_invoice.pdf"
    create_sample_invoice(sample_path)
    run_pipeline(sample_path, doc_type="invoice")