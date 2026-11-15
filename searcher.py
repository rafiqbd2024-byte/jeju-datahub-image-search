"""
searcher.py - Multimodal (CLIP) hybrid search engine

* Vision CLIP  (clip-ViT-B-32)                      -> image embeddings (512-d)
* Multilingual CLIP (clip-ViT-B-32-multilingual-v1) -> Korean/English text embeddings
  in the SAME 512-d space, so a Korean query can be compared with chart images.

Hybrid score = visual similarity (query <-> image)
             + semantic similarity (query <-> page text)
             + keyword score (query + KSIC synonym expansion <-> page text)
"""
import json
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
META_PATH = DATA_DIR / "metadata.json"
EMB_PATH = DATA_DIR / "embeddings.npz"
SYN_PATH = BASE_DIR / "industries.json"

IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# Hybrid ranking weights and calibration (raw cosine -> 0..1)
WEIGHTS = {"visual": 0.40, "semantic": 0.25, "keyword": 0.35}
CALIB = {"visual": (0.10, 0.32), "semantic": (0.20, 0.80)}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_text_model():
    """Multilingual CLIP text encoder (used at query time)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL, device="cpu")


def load_models():
    """Image + text encoders (used when building the index)."""
    from sentence_transformers import SentenceTransformer

    image_model = SentenceTransformer(IMAGE_MODEL, device="cpu")
    return image_model, load_text_model()


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #
def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build_embeddings(batch_size: int = 16):
    """Encode all page images and page texts, save precomputed cache (embeddings.npz)."""
    from PIL import Image

    records = json.loads(META_PATH.read_text(encoding="utf-8"))
    image_model, text_model = load_models()

    img_vecs = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        imgs = [Image.open(BASE_DIR / r["image"]).convert("RGB") for r in batch]
        img_vecs.append(image_model.encode(imgs, convert_to_numpy=True, show_progress_bar=False))
        print(f"  images {min(i + batch_size, len(records))}/{len(records)}")

    texts = [f"{r['title']}. {r['text'][:300]}" for r in records]
    txt_vecs = text_model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 show_progress_bar=True)

    np.savez_compressed(
        EMB_PATH,
        ids=np.array([r["id"] for r in records]),
        image_emb=_normalize(np.vstack(img_vecs)).astype(np.float16),
        text_emb=_normalize(txt_vecs).astype(np.float16),
    )
    print(f"[done] embeddings saved -> {EMB_PATH.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #
def _load_synonyms():
    try:
        data = json.loads(SYN_PATH.read_text(encoding="utf-8"))
        return {k.lower(): v for k, v in data.get("synonyms", {}).items()}
    except FileNotFoundError:
        return {}


SYNONYMS = _load_synonyms()


def _term_pattern(term: str) -> re.Pattern:
    """Word-boundary-aware regex pattern for ASCII terms; literal match for Korean/CJK."""
    escaped = re.escape(term)
    if re.search(r"^[a-z0-9]", term, flags=re.I):
        escaped = r"(?<![a-z0-9])" + escaped
    if re.search(r"[a-z0-9]$", term, flags=re.I):
        escaped = escaped + r"(?![a-z0-9])"
    return re.compile(escaped, flags=re.I)


def _key_matches_query(key: str, q: str, tokens: set) -> bool:
    """Word-boundary-aware check for whether synonym dictionary key matches query."""
    if key == q or key in tokens:
        return True
    if re.search(r"^[a-z0-9\s]+$", key, flags=re.I):
        return bool(_term_pattern(key).search(q))
    if len(key) < 2:
        return False
    return bool(re.search(rf"(?<![가-힣a-z0-9]){re.escape(key)}", q, flags=re.I))


def expand_query_terms(query: str):
    """Return [(term, weight, kind)] - original tokens + KSIC synonym expansion."""
    q = query.strip().lower()
    terms, seen = [], set()

    def add(t, w, kind):
        t = t.strip().lower()
        if t and t not in seen:
            seen.add(t)
            terms.append((t, w, kind))

    add(q, 1.0, "query")
    raw_tokens = [tok for tok in re.split(r"[\s,]+", q) if len(tok) >= 2]
    for tok in raw_tokens:
        add(tok, 1.0, "query")
    token_set = set(raw_tokens)
    for key, syns in SYNONYMS.items():
        if _key_matches_query(key, q, token_set):
            for s in syns:
                add(s, 0.7, "synonym")
    return terms


def _mask_matches(text: str, pattern: re.Pattern):
    """Count non-overlapping matches of pattern in text and replace matched spans with null chars."""
    matches = list(pattern.finditer(text))
    if not matches:
        return 0, text
    chars = list(text)
    for m in matches:
        for idx in range(m.start(), m.end()):
            chars[idx] = "\x00"
    return len(matches), "".join(chars)


def compute_keyword_score(text: str, terms, title: str = ""):
    """Keyword score in [0, 1] (saturating) and list of matched (term, count, kind)."""
    work_low, work_tlow = text.lower(), title.lower()
    matched_by_idx, raw = {}, 0.0
    # Match query terms before synonyms, and longer phrases before shorter substrings
    ordered = sorted(
        enumerate(terms),
        key=lambda item: (0 if item[1][2] == "query" else 1, -len(item[1][0])),
    )
    for idx, (term, w, kind) in ordered:
        pat = _term_pattern(term)
        c, work_low = _mask_matches(work_low, pat)
        if c:
            matched_by_idx[idx] = (term, c, kind)
            raw += w * (0.5 + 0.1 * min(c, 5))
            tc, work_tlow = _mask_matches(work_tlow, pat)
            if tc:
                raw += w * 0.8          # term appears in the chart title
    matched = [matched_by_idx[i] for i in sorted(matched_by_idx)]
    score = 1.0 - float(np.exp(-raw / 1.5))
    return score, matched


def extract_snippet(text: str, terms, width: int = 70):
    low = text.lower()
    for term, _, _ in terms:
        pos = low.find(term)
        if pos >= 0:
            s, e = max(0, pos - width), min(len(text), pos + len(term) + width)
            return ("…" if s > 0 else "") + text[s:e].replace("\n", " ") + ("…" if e < len(text) else "")
    return text[: width * 2].replace("\n", " ") + ("…" if len(text) > width * 2 else "")


def _calib(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Search engine
# --------------------------------------------------------------------------- #
class JejuMultimodalSearcher:
    def __init__(self, text_model=None):
        self.records = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.text_model = text_model
        self.image_emb = self.text_emb = None
        self.load_index()

    def load_index(self):
        if not EMB_PATH.exists():
            return
        try:
            data = np.load(EMB_PATH)
            order = {rid: i for i, rid in enumerate(data["ids"].tolist())}
            idx = [order[r["id"]] for r in self.records]
            self.image_emb = data["image_emb"].astype(np.float32)[idx]
            self.text_emb = data["text_emb"].astype(np.float32)[idx]
        except Exception as e:  # noqa: BLE001
            print(f"[warn] embeddings index mismatch or unreadable ({e}); falling back to keyword-only mode")
            self.image_emb = self.text_emb = None

    @property
    def multimodal_ready(self):
        return self.text_model is not None and self.image_emb is not None

    def _query_vector(self, query, terms):
        phrases = [query] + [t for t, _, k in terms if k == "synonym"][:3]
        vecs = self.text_model.encode(phrases, convert_to_numpy=True, show_progress_bar=False)
        vecs = _normalize(vecs)
        weights = np.array([1.0] + [0.5] * (len(phrases) - 1))[:, None]
        v = (vecs * weights).sum(0)
        return v / (np.linalg.norm(v) + 1e-12)

    def search(self, query: str, top_k: int = 40, report_filter=None):
        query = (query or "").strip()
        if not query:
            return []
        terms = expand_query_terms(query)
        n = len(self.records)

        if self.multimodal_ready:
            qv = self._query_vector(query, terms)
            vis_raw = self.image_emb @ qv
            sem_raw = self.text_emb @ qv
        else:
            vis_raw = sem_raw = np.zeros(n)
        vis = _calib(vis_raw, *CALIB["visual"])
        sem = _calib(sem_raw, *CALIB["semantic"])

        results = []
        for i, r in enumerate(self.records):
            if report_filter and r["report_id"] not in report_filter:
                continue
            kw, matched = compute_keyword_score(r["text"], terms, r.get("title", ""))
            if self.multimodal_ready:
                score = WEIGHTS["visual"] * vis[i] + WEIGHTS["semantic"] * sem[i] + WEIGHTS["keyword"] * kw
            else:
                score = kw
            results.append(
                {
                    **r,
                    "score": float(score),
                    "visual": float(vis[i]),
                    "semantic": float(sem[i]),
                    "keyword": float(kw),
                    "matched": matched,
                    "snippet": extract_snippet(r["text"], terms),
                    "terms": terms,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
