"""
services/highlighter.py

Draws highlight annotations on a PDF using PyMuPDF.
Each field gets a distinct color so you can tell them apart visually.
"""

import fitz  # PyMuPDF
import os
from typing import Dict, List


# ─── Color palette (RGB 0-1 scale) per field type ──────────────────────────────
FIELD_COLORS = {
    "invoice_number"  : (1.0, 0.9, 0.0),   # Yellow
    "invoice_date"    : (0.5, 1.0, 0.5),   # Green
    "vendor_name"     : (0.5, 0.8, 1.0),   # Blue
    "customer_name"   : (0.5, 0.8, 1.0),   # Blue
    "total_amount"    : (1.0, 0.5, 0.5),   # Red/Pink
    "tax_amount"      : (1.0, 0.7, 0.4),   # Orange
    "due_date"        : (0.8, 0.5, 1.0),   # Purple
    "name"            : (0.5, 0.8, 1.0),   # Blue
    "date"            : (0.5, 1.0, 0.5),   # Green
    "amount"          : (1.0, 0.5, 0.5),   # Red
    "id_number"       : (1.0, 0.9, 0.0),   # Yellow
    "default"         : (1.0, 0.95, 0.4),  # Light yellow fallback
}

PADDING = 2  # px padding around highlight box


def _get_color(field_name: str) -> tuple:
    return FIELD_COLORS.get(field_name, FIELD_COLORS["default"])


def highlight_pdf(
    input_path : str,
    output_path: str,
    field_coords: Dict[str, List[Dict]],
    add_label: bool = True
) -> str:
    """
    Draw highlights on the PDF for each matched field.

    Args:
        input_path  : Path to original PDF
        output_path : Path to save highlighted PDF
        field_coords: Output of coordinate_matcher.match_all_fields()
        add_label   : If True, add a small label above each highlight

    Returns:
        output_path (for chaining)
    """
    doc = fitz.open(input_path)

    for field_name, coords_list in field_coords.items():
        color = _get_color(field_name)

        for coord in coords_list:
            page_num = coord["page"]
            page     = doc[page_num]

            # Build rect with small padding
            rect = fitz.Rect(
                coord["x0"] - PADDING,
                coord["y0"] - PADDING,
                coord["x1"] + PADDING,
                coord["y1"] + PADDING
            )

            # Draw highlight annotation
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=color)
            annot.set_opacity(0.5)
            annot.update()

            # Optional: draw a colored rectangle border too
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(
                color=color,
                fill=color,
                fill_opacity=0.25,
                width=1.5
            )
            shape.commit()

            # Optional: small field label above the box
            if add_label:
                label_y = max(coord["y0"] - PADDING - 1, 5)
                page.insert_text(
                    point=fitz.Point(coord["x0"], label_y),
                    text=field_name.replace("_", " ").title(),
                    fontsize=6,
                    color=(0.2, 0.2, 0.8)
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()

    return output_path