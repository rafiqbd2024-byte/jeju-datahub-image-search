"""
app.py - Streamlit web service: Jeju Data Hub Chart & Image Search (CLIP)

Run locally:  streamlit run app.py
"""
import html
import io
import re
from pathlib import Path

import streamlit as st

from searcher import JejuMultimodalSearcher, load_text_model

BASE_DIR = Path(__file__).resolve().parent

# After deployment, put your public URL here (also used for the QR code).
APP_URL = "https://jeju-data-image-search.streamlit.app"
GITHUB_URL = "https://github.com/rafiqbd2024-byte/jeju-datahub-image-search"

QUICK_KEYWORDS = ["렌터카", "외식", "카페", "숙박", "골프장", "면세점", "관광객", "연령별 소비", "rental car", "wind farm"]

st.set_page_config(page_title="Jeju Data Hub Image Search", page_icon="🌴", layout="wide")

st.markdown(
    """
    <style>
    .score-badge {display:inline-block;padding:2px 8px;border-radius:10px;font-weight:600;
                  font-size:0.8rem;color:white;}
    .card-title {font-weight:600;font-size:0.92rem;margin:4px 0 2px 0;line-height:1.25;}
    .card-meta {color:#6b7280;font-size:0.78rem;}
    .card-reason {font-size:0.78rem;background:#f1f5f9;border-radius:6px;padding:4px 6px;margin-top:4px;}
    .card-snippet {font-size:0.78rem;color:#374151;margin-top:4px;}
    mark {background:#fde68a;padding:0 2px;border-radius:3px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="AI 모델 로딩 중 (Loading multilingual CLIP)…")
def get_searcher():
    try:
        model = load_text_model()
    except Exception as e:  # noqa: BLE001
        st.session_state["model_error"] = str(e)
        model = None
    return JejuMultimodalSearcher(text_model=model)


@st.cache_data
def read_bytes(rel_path: str) -> bytes:
    return (BASE_DIR / rel_path).read_bytes()


@st.cache_data
def make_qr(url: str) -> bytes:
    import qrcode

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def highlight_text(text: str, terms) -> str:
    safe = html.escape(text)
    words = sorted({t for t, _, _ in terms if len(t) >= 2}, key=len, reverse=True)
    for w in words:
        safe = re.sub(re.escape(html.escape(w)), lambda m: f"<mark>{m.group(0)}</mark>", safe, flags=re.I)
    return safe


def get_match_reason(r) -> str:
    parts = []
    q_hits = [f"'{t}'({c})" for t, c, k in r["matched"] if k == "query"]
    s_hits = [f"'{t}'" for t, c, k in r["matched"] if k == "synonym"]
    if q_hits:
        parts.append("키워드 일치 " + ", ".join(q_hits[:3]))
    if s_hits:
        parts.append("동의어/산업분류 확장 " + ", ".join(s_hits[:3]))
    if r["visual"] > 0:
        parts.append(f"이미지 유사도 {r['visual']*100:.0f}%")
    if r["semantic"] > 0:
        parts.append(f"본문 의미 유사도 {r['semantic']*100:.0f}%")
    return " · ".join(parts) if parts else "시각적 특징 기반 유사 결과"


def score_color(p: float) -> str:
    return "#16a34a" if p >= 70 else "#2563eb" if p >= 50 else "#f59e0b" if p >= 30 else "#9ca3af"


@st.dialog("이미지 상세 보기 (Detail view)", width="large")
def show_image_modal(r):
    st.image(str(BASE_DIR / r["image"]), use_container_width=True)
    st.markdown(f"**{r['title']}**  \n{r['report_title']} · p.{r['page']} / {r['n_pages']}")
    if "score" in r:
        st.markdown(f"**일치도 (Match): {r['score']*100:.1f}%** — {get_match_reason(r)}")
    c1, c2 = st.columns(2)
    c1.download_button("🖼️ 이미지 다운로드 (PNG)", read_bytes(r["image"]),
                       file_name=Path(r["image"]).name, mime="image/png", use_container_width=True)
    c2.download_button("📄 원본 리포트 (PDF)", read_bytes(r["pdf"]),
                       file_name=Path(r["pdf"]).name, mime="application/pdf", use_container_width=True)
    with st.expander("페이지 텍스트 (Extracted text)"):
        st.write(r["text"][:1500])


def render_grid(results, n_cols=4, show_score=True):
    cols = st.columns(n_cols)
    for i, r in enumerate(results):
        with cols[i % n_cols]:
            with st.container(border=True):
                st.image(str(BASE_DIR / r["image"]), use_container_width=True)
                if show_score:
                    p = r["score"] * 100
                    st.markdown(f"<span class='score-badge' style='background:{score_color(p)}'>"
                                f"일치도 {p:.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-title'>{html.escape(r['title'])}</div>"
                            f"<div class='card-meta'>{html.escape(r['report_title'])} · p.{r['page']}</div>",
                            unsafe_allow_html=True)
                if show_score:
                    st.markdown(f"<div class='card-reason'>💡 {html.escape(get_match_reason(r))}</div>"
                                f"<div class='card-snippet'>{highlight_text(r['snippet'], r['terms'])}</div>",
                                unsafe_allow_html=True)
                if st.button("🔍 크게 보기", key=f"view_{r['id']}", use_container_width=True):
                    show_image_modal(r)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
searcher = get_searcher()
records = searcher.records
reports = sorted({r["report_id"] for r in records})

with st.sidebar:
    st.header("⚙️ 검색 설정 (Settings)")
    top_k = st.slider("최대 결과 수 (Max results)", 8, 80, 40, step=4)
    report_filter = st.multiselect("리포트 필터 (Filter by report)", reports)
    n_cols = st.select_slider("열 개수 (Columns)", options=[2, 3, 4, 5], value=4)
    st.divider()
    st.caption(f"📚 리포트 {len(reports)}개 · 🖼️ 이미지 {len(records)}개")
    st.caption("Model: Multilingual CLIP (ViT-B/32) + KSIC synonym hybrid ranking")
    st.divider()
    st.subheader("📱 모바일로 접속 (QR)")
    try:
        st.image(make_qr(APP_URL), width=170)
    except Exception:  # noqa: BLE001
        pass
    st.caption(APP_URL)
    st.markdown(f"[GitHub Repository]({GITHUB_URL})")

st.markdown("<h2 style='text-align:center;color:#2563eb'>🌴 제주데이터허브 차트 & 이미지 검색</h2>"
            "<p style='text-align:center;color:#6b7280'>Jeju Data Hub Chart & Image Search · "
            "키워드를 입력하면 리포트 속 차트와 인포그래픽을 AI(CLIP)가 즉시 찾아드립니다.</p>",
            unsafe_allow_html=True)

if not searcher.multimodal_ready:
    st.warning("AI 임베딩 모델을 불러오지 못해 키워드 검색 모드로 동작합니다. "
               "(Running in keyword-only mode.)")

if "query" not in st.session_state:
    st.session_state.query = ""

qcols = st.columns(len(QUICK_KEYWORDS) + 1)
if qcols[0].button("전체보기", use_container_width=True):
    st.session_state.query = ""
    st.session_state.show_all = True
for c, kw in zip(qcols[1:], QUICK_KEYWORDS):
    if c.button(kw, use_container_width=True):
        st.session_state.query = kw
        st.session_state.show_all = False

query = st.text_input("검색어 (Search keyword)", key="query",
                      placeholder="예: 렌터카, 외식, 골프장 매출, 관광객 소비, rental car … (Enter)")

if query.strip():
    st.session_state.show_all = False
    with st.spinner("검색 중…"):
        results = searcher.search(query, top_k=top_k, report_filter=report_filter or None)

    head, slider_col = st.columns([3, 2])
    with slider_col:
        threshold = st.slider("최소 일치도 필터 (Similarity threshold, %)", 0, 100, 30, step=5,
                              help="이 값보다 일치도가 낮은 결과는 숨깁니다. (Hide results below this score)")
    shown = [r for r in results if r["score"] * 100 >= threshold]
    with head:
        st.markdown(f"### 🔍 '{html.escape(query)}' 검색 결과")
        best = results[0]["score"] * 100 if results else 0
        st.caption(f"최고 일치도: {best:.1f}% | 표시: {len(shown)}개 / 전체 후보 {len(results)}개 "
                   f"| 임계값 {threshold}% 이상")
    expanded = [t for t, _, k in results[0]["terms"] if k == "synonym"] if results else []
    if expanded:
        st.caption("🔁 동의어 확장 (Query expansion): " + ", ".join(expanded[:8]))
    if shown:
        render_grid(shown, n_cols=n_cols)
    else:
        st.info("임계값을 만족하는 결과가 없습니다. 슬라이더를 낮춰 보세요. (Lower the threshold.)")
elif st.session_state.get("show_all"):
    pool = [r for r in records if not report_filter or r["report_id"] in report_filter]
    st.markdown(f"### 🗂️ 전체 이미지 ({len(pool)})")
    render_grid(pool[:top_k], n_cols=n_cols, show_score=False)
else:
    st.info("위에서 키워드를 입력하거나 빠른 검색 버튼을 눌러 보세요. "
            "(Type a keyword or click a quick-search button.)")
