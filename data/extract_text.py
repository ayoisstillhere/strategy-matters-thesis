# data/extract_text.py
import re
import pdfplumber
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "wahlprogramme" / "raw"
TEXT_DIR = DATA_DIR / "wahlprogramme" / "text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

CID_RE = re.compile(r"\(cid:\d+\)")
BLANK_LINES_RE = re.compile(r"\n{3,}")
SINGLE_CHAR_LINES_RE = re.compile(r"(?m)^.{1,2}\n(?=.{1,2}\n)")

# Number of cover pages to skip per PDF (decorative layouts garble extraction)
SKIP_PAGES = {
    "afd_2025.pdf": 1,
    "cdu_csu_2025.pdf": 1,
    "fdp_2025.pdf": 1,
    "gruene_2025.pdf": 1,
    "linke_2025.pdf": 1,
    "spd_2025.pdf": 1,
}

# Lines to strip from start of extracted text (residual garbled fragments)
STRIP_PATTERNS = {
    "afd_2025.pdf": re.compile(
        r"^.*?(?=Das vorliegende Wahlprogramm)", re.DOTALL
    ),
    "linke_2025.pdf": re.compile(
        r"^.*?(?=Alle wollen regieren\. Wir wollen verändern\.)", re.DOTALL
    ),
}

for pdf_path in RAW_DIR.glob("*.pdf"):
    skip = SKIP_PAGES.get(pdf_path.name, 0)
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages[skip:]:
            cleaned = page.dedupe_chars(tolerance=1)
            pages.append(cleaned.extract_text() or "")
        text = "\n\n".join(pages)
    text = CID_RE.sub("", text)
    text = SINGLE_CHAR_LINES_RE.sub("", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    if pdf_path.name in STRIP_PATTERNS:
        text = STRIP_PATTERNS[pdf_path.name].sub("", text)
    text = text.strip()
    out_path = TEXT_DIR / pdf_path.with_suffix(".txt").name
    out_path.write_text(text, encoding="utf-8")
    print(f"{pdf_path.name} → {len(text):,} chars")