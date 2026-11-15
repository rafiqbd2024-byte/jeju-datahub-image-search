"""
run_pipeline.py - Runs the full offline pipeline (run once on a local PC)

  1단계 collector.py : collect report PDFs           -> data/reports/
  2단계 extractor.py : page text + 150 DPI PNG images -> data/images/, data/metadata.json
  3단계 searcher.py  : CLIP embeddings                 -> data/embeddings.npz

After this, commit the data/ folder to GitHub; the Streamlit app (app.py)
only loads the precomputed index, so the cloud server stays light.
"""
import time

from collector import fetch_and_download_reports
from extractor import extract_pages_and_render_images
from searcher import build_embeddings


def main():
    t0 = time.time()
    print("=== Step 1/3: Collect reports ===")
    fetch_and_download_reports()
    print("\n=== Step 2/3: Extract pages and render images ===")
    extract_pages_and_render_images()
    print("\n=== Step 3/3: Build CLIP embeddings ===")
    build_embeddings()
    print(f"\nPipeline finished in {time.time() - t0:.1f} s. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
