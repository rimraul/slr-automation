"""
Lean prototype: render the 4 no-PDF pilot records' extracted .txt files into
plain PDFs, so FT_SCREENING's docling parser (which hard-requires a .pdf
file -- schema.py:285 `pdf_path: Path`, no plain-text input path exists)
can screen them too. These 4 have real full text on disk already (Elsevier
Article Retrieval API XML-to-text for 3, a manually recovered .docx for the
4th) -- they were just never saved as an actual PDF file.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).parent
PILOT_DIR = HERE / "fulltext_pilot"
TEXT_DIR = PILOT_DIR / "text"
PDF_DIR = PILOT_DIR / "pdfs"

REC_IDS = [
    "wos-a799cafa17fa9259a377",
    "wos-9140e6832d04f557e051",
    "wos-d02908a43cfd326b63ab",
    "wos-0be9828c8c8318cd9b72",
]


def safe_id(rec_id: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", rec_id)


def main() -> None:
    for rec_id in REC_IDS:
        sid = safe_id(rec_id)
        text_path = TEXT_DIR / f"{sid}.txt"
        pdf_path = PDF_DIR / f"{sid}.pdf"
        text = text_path.read_text(encoding="utf-8", errors="replace")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        for line in text.splitlines():
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            if not safe_line.strip():
                pdf.ln(5)
                continue
            for wrapped in textwrap.wrap(safe_line, width=70, break_long_words=True) or [""]:
                pdf.multi_cell(0, 5, wrapped, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.output(str(pdf_path))

        # fpdf2 writes /MediaBox only on the Pages parent (inherited, valid
        # PDF spec) -- Docling's pypdfium2-based backend does not resolve
        # inherited page attributes and rejects such pages with "could not
        # find the page-dimensions". Round-trip through pypdf to force an
        # explicit /MediaBox onto every page object.
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        for page in reader.pages:
            page.mediabox = page.mediabox
            writer.add_page(page)
        with open(pdf_path, "wb") as fh:
            writer.write(fh)

        print(f"wrote {pdf_path}  ({pdf_path.stat().st_size} bytes, from {len(text)} chars)")


if __name__ == "__main__":
    main()
