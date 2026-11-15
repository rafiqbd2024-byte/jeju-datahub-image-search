# 🌴 Jeju Data Hub Chart & Image Search (CLIP)

An AI multimodal image search service for the **Jeju Data Hub** (https://jejudatahub.net).
Type a keyword in Korean or English (e.g. `렌터카`, `외식`, `골프장`, `rental car`). The app finds the matching
charts and infographics inside the Jeju Data Hub periodic reports, so you don't have to open each PDF.

*Advanced Management Information Systems: Midterm Exam Replacement Project (Jeju National University)*

## Features
- **Cross-modal search**: Multilingual CLIP turns a Korean or English query into a vector in the same 512-dimensional space as the chart images.
- **Hybrid ranking**: combines visual similarity (query ↔ image), semantic similarity (query ↔ page text) and keyword matching.
- **Query expansion**: everyday terms are mapped to official industry names based on KSIC (`industries.json`), e.g. *렌터카 → 자동차 임대업*, *외식 → 음식점업*.
- **Explainable results**: each result card shows the match score (%), why it matched, and a highlighted text snippet.
- **Similarity threshold slider**: filters the results in real time.
- Modal detail view, PNG image download, original PDF download, and a QR code for mobile access.

## Architecture
```
collector.py  → data/reports/*.pdf            (1. ingestion)
extractor.py  → data/images/*.png (150 DPI)    (2. ETL: page text + rendering)
                data/metadata.json
searcher.py   → data/embeddings.npz            (3. CLIP embeddings, precomputed)
app.py        → Streamlit web dashboard        (4. search UI, cloud deployment)
run_pipeline.py runs steps 1–3 in order.
```
Models: `clip-ViT-B-32` (image encoder) and `sentence-transformers/clip-ViT-B-32-multilingual-v1` (text encoder).

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# put 5-10 Jeju Data Hub report PDFs into data/reports/
python run_pipeline.py
streamlit run app.py
```

## Deploy
1. Push this repository, including the `data/` folder, to a public GitHub repo.
2. On https://share.streamlit.io, choose **Create app**, select the repo, set the main file to `app.py`, and click **Deploy**.

All paths are relative (`data/...`), so the same code runs both locally and on Streamlit Community Cloud.
