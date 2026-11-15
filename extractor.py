"""
extractor.py - Stage 1-2: Extract page text and render high-resolution chart images

For every page of every report in data/reports/:
  * extract the page-level text (used for keyword matching and snippets)
  * render the page as a PNG image (150 DPI) into data/images/

Output: data/images/*.png and data/metadata.json
"""
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = DATA_DIR / "reports"
IMAGE_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.json"

DPI = 150
MIN_TEXT_CHARS = 15        # skip almost empty pages
MIN_INK_RATIO = 0.01       # skip blank (white) pages


def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _guess_title(text: str, fallback: str) -> str:
    """Pick the first meaningful line as the chart/page title."""
    for line in text.splitlines():
        line = line.strip()
        if 4 <= len(line) <= 60 and not re.fullmatch(r"[\d\s.\-/%()]+", line):
            return line
    return fallback


def _ink_ratio(img: Image.Image) -> float:
    gray = img.convert("L").resize((200, int(200 * img.height / img.width)))
    hist = gray.histogram()
    dark = sum(hist[:230])
    return dark / max(1, sum(hist))


def extract_pages_and_render_images(dpi: int = DPI):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(REPORT_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDF in data/reports/. Run collector.py first.")

    rmeta = {}
    meta_file = DATA_DIR / "reports_meta.json"
    if meta_file.exists():
        rmeta = {m["report_id"]: m for m in json.loads(meta_file.read_text(encoding="utf-8"))}

    records = []
    scale = dpi / 72
    for pdf_path in pdfs:
        report_id = pdf_path.stem
        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        texts = [_clean(pdf[i].get_textpage().get_text_range()) for i in range(n_pages)]
        # lines repeated on many pages = running header/footer -> not a good title
        from collections import Counter
        norm = lambda l: re.sub(r"\d+", "#", l.strip())
        freq = Counter(norm(l) for t in texts for l in set(t.splitlines()))
        repeated = {l for l, c in freq.items() if n_pages >= 3 and c >= max(3, n_pages * 0.4)}
        kept = 0
        for idx in range(n_pages):
            page = pdf[idx]
            text = texts[idx]
            title_src = "\n".join(l for l in text.splitlines() if norm(l) not in repeated)
            img = page.render(scale=scale).to_pil().convert("RGB")
            if len(text) < MIN_TEXT_CHARS and _ink_ratio(img) < MIN_INK_RATIO:
                continue  # blank page
            img_name = f"{report_id}_p{idx + 1}.png"
            img_path = IMAGE_DIR / img_name
            img.save(img_path, optimize=True)
            records.append(
                {
                    "id": f"{report_id}_p{idx + 1}",
                    "report_id": report_id,
                    "report_title": rmeta.get(report_id, {}).get("title", report_id.replace("_", " ")),
                    "date": rmeta.get(report_id, {}).get("date", ""),
                    "source": rmeta.get(report_id, {}).get("source", ""),
                    "page": idx + 1,
                    "n_pages": n_pages,
                    "title": _guess_title(title_src, f"{report_id} p.{idx + 1}"),
                    "image": f"data/images/{img_name}",      # relative paths
                    "pdf": f"data/reports/{pdf_path.name}",
                    "text": text[:4000],
                }
            )
            kept += 1
        pdf.close()
        print(f"[ok] {pdf_path.name}: {kept}/{n_pages} pages rendered")

    META_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(records)} page images -> {META_PATH.relative_to(BASE_DIR)}")
    return records


if __name__ == "__main__":
    extract_pages_and_render_images()
