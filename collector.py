"""
collector.py - Stage 1-1: Collect Jeju Data Hub periodic reports (PDF)

Two ways to collect reports:
  1) Put URLs in `report_urls.txt` (one per line: URL [TAB or |] optional title)
     and run this script - the PDFs are downloaded into data/reports/.
  2) Download the PDFs manually from https://jejudatahub.net/ (리포트 > 정기리포트)
     and save them into data/reports/. The script then just registers them.

Output: data/reports/*.pdf and data/reports_meta.json
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = DATA_DIR / "reports"
META_PATH = DATA_DIR / "reports_meta.json"
URL_LIST = BASE_DIR / "report_urls.txt"

HEADERS = {"User-Agent": "Mozilla/5.0 (JejuDataHub-ImageSearch student project)"}


def _safe_name(name: str) -> str:
    name = re.sub(r"\.pdf$", "", name.strip(), flags=re.I)
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return name[:120] or "report"


def _read_url_list():
    items = []
    if not URL_LIST.exists():
        return items
    for line in URL_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t|\|", line, maxsplit=1)
        url = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else ""
        items.append((url, title))
    return items


API = "https://jejudatahub.net/api"


def fetch_from_jejudatahub_api(max_reports: int = 10):
    """Jeju Data Hub open endpoints: report list -> report view -> file download."""
    lst = requests.get(f"{API}/report/list", headers=HEADERS, timeout=60, params={
        "selection": "title + contents", "keyword": "", "isPaging": "true", "length": max_reports}).json()
    meta = {}
    for item in lst.get("data", [])[:max_reports]:
        view = requests.get(f"{API}/report/view", params={"id": item["id"], "user": ""},
                            headers=HEADERS, timeout=60).json()
        file = view.get("file") or {}
        for info in file.get("fileInfos", []):
            if not info["fileName"].lower().endswith(".pdf"):
                continue
            stem = _safe_name(f"{item['id']}_{Path(info['fileName']).stem}")
            target = REPORT_DIR / f"{stem}.pdf"
            meta[target.stem] = {"title": view["title"], "date": view.get("createAt", ""),
                                 "source": f"https://jejudatahub.net/report/serial/view/{item['id']}"}
            if target.exists():
                print(f"[skip] {target.name}")
                continue
            r = requests.get(f"{API}/file/test-download", params={"id": info["id"], "fileId": file["id"]},
                             headers=HEADERS, timeout=180)
            if r.ok and r.content.startswith(b"%PDF"):
                target.write_bytes(r.content)
                print(f"[ok] {target.name} ({len(r.content)//1024} KB)")
            else:
                print(f"[warn] download failed: {view['title']}")
            break
    return meta


def fetch_and_download_reports(max_reports: int = 10):
    """Download the latest periodic reports (API, or report_urls.txt) and register all PDFs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    existing_meta = {}
    if META_PATH.exists():
        try:
            for entry in json.loads(META_PATH.read_text(encoding="utf-8")):
                if "report_id" in entry:
                    existing_meta[entry["report_id"]] = entry
        except Exception:  # noqa: BLE001
            pass

    api_meta = {}
    try:
        api_meta = fetch_from_jejudatahub_api(max_reports)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Jeju Data Hub API not reachable ({e}); using report_urls.txt / local PDFs")

    url_meta = {}
    for i, (url, title) in enumerate(_read_url_list()[:max_reports], start=1):
        stem = _safe_name(title or Path(url.split("?")[0]).stem or f"report_{i}")
        target = REPORT_DIR / f"{stem}.pdf"
        url_meta[target.stem] = {"title": title, "source": url}
        if target.exists():
            print(f"[skip] already downloaded: {target.name}")
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                print(f"[warn] not a PDF response: {url}")
                continue
            target.write_bytes(r.content)
            print(f"[ok] downloaded: {target.name} ({len(r.content)//1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"[error] {url}: {e}")

    pdfs = sorted(REPORT_DIR.glob("*.pdf"))
    if not pdfs:
        print(
            "\nNo PDF found in data/reports/.\n"
            "Download 5-10 periodic reports from https://jejudatahub.net/ (리포트 > 정기리포트)\n"
            "and save them into data/reports/, or list their URLs in report_urls.txt."
        )
        sys.exit(1)

    meta = []
    for p in pdfs:
        prev = existing_meta.get(p.stem, {})
        m = api_meta.get(p.stem, {})
        u = url_meta.get(p.stem, {})
        meta.append(
            {
                "report_id": p.stem,
                "title": m.get("title") or u.get("title") or prev.get("title") or p.stem.replace("_", " "),
                "date": m.get("date") or prev.get("date", ""),
                "source": m.get("source") or prev.get("source") or u.get("source", ""),
                "file": f"data/reports/{p.name}",  # relative path (cloud compatible)
                "size_kb": p.stat().st_size // 1024,
                "collected_at": m.get("collected_at") or prev.get("collected_at") or datetime.now().strftime("%Y-%m-%d"),
            }
        )
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(meta)} reports registered -> {META_PATH.relative_to(BASE_DIR)}")
    return meta


if __name__ == "__main__":
    fetch_and_download_reports()
