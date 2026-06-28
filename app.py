"""
app.py  ·  MNIST Canvas
────────────────────────
Flow:
  1. Welcome screen   → user enters their name
  2. Guided session   → user draws digits 0,1,2,...,9 IN ORDER, one at a
                         time. Each digit is auto-saved the moment it's
                         processed (label is never ambiguous — it's
                         always whatever digit the session is currently
                         on), then the session auto-advances to the next
                         digit.
  3. Session complete → shown after digit 9 is saved; option to start
                         a brand new 0-9 session.
  4. Dataset tab       → browse, download CSV / NPY (reads from the
                         shared Google Sheet).

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import logging
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MNIST Canvas",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from mnist_processor import MNISTProcessor
from ui_components import (
    render_pixel_matrix,
    render_mnist_image,
    render_stats,
    render_download_buttons,
    render_pipeline_steps,
)
from sheets_manager import (
    save_sample,
    dataset_summary,
    total_samples,
    get_npy_bytes,
    get_csv_bytes,
    get_grid_csv_bytes,
    image_to_grid_csv_bytes,
    load_csv_as_records,
    clear_sheet_cache,
)

logging.basicConfig(level=logging.INFO)

# ── modern light palette ─────────────────────────────────────────────────────
_BG       = "#FFFFFF"
_SURFACE  = "#F7F8FA"
_CARD     = "#FFFFFF"
_ACCENT   = "#6366F1"   # indigo
_ACCENT2  = "#EC4899"   # pink
_TEXT     = "#111827"
_MUTED    = "#6B7280"
_BORDER   = "#E5E7EB"
_GREEN    = "#10B981"

TOTAL_DIGITS = 10  # session covers 0..9


# ─────────────────────────────────────────────────────────────────────────────
# CSS — modern SaaS-style light theme
# ─────────────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [data-testid="stAppViewContainer"] {{
          background-color: {_SURFACE};
          color: {_TEXT};
          font-family: 'Inter', sans-serif;
      }}
      [data-testid="stHeader"] {{ background: transparent; }}
      * {{ font-family: 'Inter', sans-serif; }}

      [data-testid="stSidebar"] {{
          background-color: {_CARD};
          border-right: 1px solid {_BORDER};
      }}
      [data-testid="stSidebar"] * {{ color: {_TEXT}; }}

      h1, h2, h3 {{
          font-family: 'Inter', sans-serif;
          font-weight: 800;
          letter-spacing: -0.02em;
          color: {_TEXT};
      }}

      .ui-card {{
          background: {_CARD};
          border: 1px solid {_BORDER};
          border-radius: 16px;
          padding: 1.4rem 1.5rem;
          box-shadow: 0 1px 3px rgba(17,24,39,0.04), 0 1px 2px rgba(17,24,39,0.03);
          margin-bottom: 1rem;
      }}

      [data-testid="stMetric"] {{
          background: {_SURFACE};
          border: 1px solid {_BORDER};
          border-radius: 12px;
          padding: 14px 16px;
      }}
      [data-testid="stMetricValue"] {{
          color: {_TEXT}; font-weight: 700; font-size: 1.15rem;
      }}
      [data-testid="stMetricLabel"] {{ color: {_MUTED}; font-size: 0.78rem; }}

      .section-label {{
          font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
          color: {_MUTED}; font-weight: 600; margin-bottom: 0.5rem;
      }}

      .welcome-wrap {{
          display: flex; align-items: center; justify-content: center;
          min-height: 70vh;
      }}
      .welcome-card {{
          background: {_CARD}; border: 1px solid {_BORDER}; border-radius: 24px;
          padding: 3rem 2.5rem; max-width: 440px; width: 100%;
          box-shadow: 0 20px 50px -12px rgba(99,102,241,0.18), 0 4px 12px rgba(17,24,39,0.04);
          text-align: center;
      }}
      .welcome-icon {{
          width: 56px; height: 56px; margin: 0 auto 1.2rem;
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2});
          border-radius: 16px; display: flex; align-items: center;
          justify-content: center; font-size: 1.6rem;
      }}
      .welcome-title {{ font-size: 1.7rem; font-weight: 800; color: {_TEXT}; margin-bottom: 0.3em; }}
      .welcome-sub {{ font-size: 0.9rem; color: {_MUTED}; margin-bottom: 1.8rem; line-height: 1.5; }}

      .saved-badge {{
          background: {_GREEN}14; border: 1px solid {_GREEN}55;
          color: #047857; border-radius: 10px; padding: 0.55rem 1rem;
          font-size: 0.85rem; font-weight: 600;
          display: inline-flex; align-items: center; gap: 0.4rem;
      }}

      [data-testid="stDataFrame"] {{
          background: {_CARD}; border: 1px solid {_BORDER}; border-radius: 12px;
      }}

      hr {{ border-color: {_BORDER}; margin: 1.2rem 0; }}

      [data-testid="stDownloadButton"] button, .stButton button {{
          border-radius: 10px; font-weight: 600; font-size: 0.86rem;
          border: 1px solid {_BORDER}; transition: all .15s ease;
      }}
      [data-testid="stDownloadButton"] button {{ background: {_CARD}; color: {_TEXT}; }}
      [data-testid="stDownloadButton"] button:hover {{ border-color: {_ACCENT}; color: {_ACCENT}; }}
      button[kind="primary"] {{
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2}) !important;
          border: none !important; color: white !important;
          box-shadow: 0 4px 14px rgba(99,102,241,0.3);
      }}
      button[kind="primary"]:hover {{ opacity: 0.92; }}

      [data-baseweb="tab-list"] {{ gap: 4px; }}
      [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; font-weight: 600; }}

      [data-testid="stSlider"] [role="slider"] {{ background-color: {_ACCENT} !important; }}

      .ui-card canvas {{
          border-radius: 12px;
          box-shadow: 0 0 0 1px {_BORDER};
      }}

      /* ── giant target-digit display ── */
      .target-digit-wrap {{
          display: flex; align-items: center; gap: 1rem;
          margin-bottom: 1rem;
      }}
      .target-digit-circle {{
          width: 84px; height: 84px; border-radius: 20px;
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2});
          display: flex; align-items: center; justify-content: center;
          font-size: 2.6rem; font-weight: 800; color: white; flex-shrink: 0;
          box-shadow: 0 8px 20px rgba(99,102,241,0.35);
      }}
      .target-digit-meta {{ font-size: 0.85rem; color: {_MUTED}; }}
      .target-digit-meta b {{ color: {_TEXT}; }}

      /* ── progress dots ── */
      .progress-dots {{ display: flex; gap: 6px; margin-bottom: 1.2rem; flex-wrap: wrap; }}
      .progress-dot {{
          width: 30px; height: 30px; border-radius: 8px;
          display: flex; align-items: center; justify-content: center;
          font-size: 0.78rem; font-weight: 700; font-family: monospace;
          border: 1.5px solid {_BORDER}; color: {_MUTED}; background: {_SURFACE};
      }}
      .progress-dot.done {{
          background: {_GREEN}1A; border-color: {_GREEN}77; color: #047857;
      }}
      .progress-dot.current {{
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2});
          border-color: transparent; color: white;
          box-shadow: 0 4px 10px rgba(99,102,241,0.4);
      }}

      /* ── session-complete card ── */
      .complete-icon {{
          width: 64px; height: 64px; margin: 0 auto 1rem;
          background: linear-gradient(135deg, {_GREEN}, #34D399);
          border-radius: 18px; display: flex; align-items: center;
          justify-content: center; font-size: 2rem;
      }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE on caching: sheets_manager.py already caches the underlying Google
# Sheets read internally (a single _st_cached_fetch layer, TTL=10s) and
# exposes a clear_cache() function to invalidate it after a save. app.py
# calls the plain dataset_summary() / total_samples() / etc. functions
# directly — wrapping them in a SECOND, independent st.cache_data layer
# here previously caused the dashboard to show stale/inconsistent data,
# since the two cache layers had different TTLs and invalidation timing.
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar(user_name: str) -> dict:
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:0.8rem 0 0.6rem'>
          <div style='width:44px;height:44px;margin:0 auto 0.6rem;
                      background:linear-gradient(135deg,{_ACCENT},{_ACCENT2});
                      border-radius:12px;display:flex;align-items:center;
                      justify-content:center;font-size:1.3rem;'>✏️</div>
          <span style='font-size:1.05rem;font-weight:800;color:{_TEXT};'>
            MNIST Canvas
          </span><br>
          <span style='font-size:0.72rem;color:{_MUTED};'>
            👤 {user_name}
          </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Change name / restart session", use_container_width=True):
            for k in ["user_name", "screen", "last_result", "session_index"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()
        st.markdown("**Canvas**")
        stroke_width = st.slider("Stroke width (px)", 8, 40, 22)
        stroke_color = st.color_picker("Stroke colour", "#FFFFFF")
        canvas_size  = st.selectbox("Canvas size", [280, 400, 560], index=0)

        st.divider()
        st.markdown("**Preprocessing**")
        show_steps  = st.toggle("Show pipeline steps", value=False)
        show_matrix = st.toggle("Show pixel matrix",   value=False)

        st.divider()
        summary = dataset_summary()
        st.markdown(f"**Dataset** · {summary['total']} samples")
        if summary["per_digit"]:
            for d, cnt in summary["per_digit"].items():
                bar = "▰" * min(cnt, 16)
                st.markdown(
                    f"<span style='font-family:monospace;font-size:0.72rem;"
                    f"color:{_MUTED}'>{d} </span>"
                    f"<span style='color:{_ACCENT};font-size:0.72rem'>{bar}</span>"
                    f"<span style='color:{_MUTED};font-size:0.72rem'> {cnt}</span>",
                    unsafe_allow_html=True,
                )

    return {
        "stroke_width": stroke_width,
        "stroke_color": stroke_color,
        "canvas_size":  canvas_size,
        "show_steps":   show_steps,
        "show_matrix":  show_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1 — Welcome / Name Gate
# ─────────────────────────────────────────────────────────────────────────────

def _screen_welcome() -> None:
    st.markdown(f"""
    <div class="welcome-wrap">
      <div class="welcome-card">
        <div class="welcome-icon">✏️</div>
        <div class="welcome-title">MNIST Canvas</div>
        <div class="welcome-sub">
          You'll be asked to draw each digit from 0 to 9, one at a time.<br>
          Every drawing is saved automatically to a shared dataset.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        name = st.text_input(
            "Your name", placeholder="e.g. Ronit",
            label_visibility="collapsed", key="name_input",
        )
        if st.button("Start session  →", type="primary", use_container_width=True):
            name = name.strip()
            if not name:
                st.error("Please enter your name to continue.")
            elif len(name) < 2:
                st.error("Name must be at least 2 characters.")
            else:
                st.session_state["user_name"] = name
                st.session_state["screen"] = "canvas"
                st.session_state["session_index"] = 0
                st.rerun()

        st.markdown(
            f"<div style='text-align:center;color:{_MUTED};font-size:0.78rem;"
            f"margin-top:1rem'>{total_samples()} drawings saved so far</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Progress dots + target digit header
# ─────────────────────────────────────────────────────────────────────────────

def _render_progress(session_index: int) -> None:
    dots_html = "<div class='progress-dots'>"
    for d in range(TOTAL_DIGITS):
        if d < session_index:
            cls = "done"
        elif d == session_index:
            cls = "current"
        else:
            cls = ""
        dots_html += f"<div class='progress-dot {cls}'>{d}</div>"
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)


def _render_target_digit(session_index: int) -> None:
    st.markdown(f"""
    <div class="target-digit-wrap">
      <div class="target-digit-circle">{session_index}</div>
      <div class="target-digit-meta">
        Draw this digit · <b>{session_index + 1} of {TOTAL_DIGITS}</b> in your session
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Screen 3 — Session complete
# ─────────────────────────────────────────────────────────────────────────────

def _screen_session_complete(user_name: str) -> None:
    st.markdown(f"""
    <div class="welcome-wrap">
      <div class="welcome-card">
        <div class="complete-icon">🎉</div>
        <div class="welcome-title">Session complete!</div>
        <div class="welcome-sub">
          Thanks <b style='color:{_ACCENT}'>{user_name}</b> — you drew all 10
          digits (0–9) and every one was saved to the shared dataset.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        if st.button("Start a new session  →", type="primary", use_container_width=True):
            st.session_state["session_index"] = 0
            st.session_state.pop("last_result", None)
            st.rerun()
        if st.button("View collected dataset", use_container_width=True):
            st.session_state["screen"] = "dataset_only"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Results renderer (shown after each digit is saved)
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(user_name: str, cfg: dict, result: dict) -> None:
    proc      = result["proc"]
    mnist_img = result["mnist_img"]
    saved_as  = result["saved_as"]

    stats     = proc.get_stats()
    png_bytes = proc.to_png_bytes()
    npy_bytes = proc.to_numpy_bytes()

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    img_col, dl_col = st.columns([1, 1], gap="medium")

    with img_col:
        st.markdown("<div class='section-label'>MNIST output</div>", unsafe_allow_html=True)
        render_mnist_image(mnist_img)

    with dl_col:
        st.markdown("<div class='section-label'>Download image</div>", unsafe_allow_html=True)
        if png_bytes and npy_bytes:
            render_download_buttons(png_bytes, npy_bytes)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Image stats</div>", unsafe_allow_html=True)
        render_stats(stats)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    save_col, _spacer = st.columns([1, 1], gap="medium")
    with save_col:
        st.markdown(
            f"<div class='saved-badge'>✓ Saved as digit {saved_as}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Saved to the shared Google Sheet dataset.")
        grid_bytes = image_to_grid_csv_bytes(mnist_img)
        st.download_button(
            "⬇ Download 28×28 grid CSV",
            grid_bytes, f"digit_{saved_as}.csv", "text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if cfg["show_steps"]:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Preprocessing pipeline</div>", unsafe_allow_html=True)
        render_pipeline_steps(proc)
        st.markdown('</div>', unsafe_allow_html=True)

    if cfg["show_matrix"]:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Pixel matrix (28×28 · uint8)</div>", unsafe_allow_html=True)
        render_pixel_matrix(mnist_img)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tab/screen
# ─────────────────────────────────────────────────────────────────────────────

def _tab_dataset() -> None:
    import pandas as pd

    st.markdown(
        f"<h3 style='color:{_TEXT};margin-bottom:0.2rem'>📊 Collected Dataset</h3>",
        unsafe_allow_html=True,
    )

    summary = dataset_summary()
    n = summary["total"]

    if n == 0:
        st.info("No samples saved yet. Draw and save some digits first!")
        return

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Total samples", n)
    m2.metric("Contributors", len(summary["contributors"]))
    st.markdown('</div>', unsafe_allow_html=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F8FA")

    digits = list(range(10))
    counts = [summary["per_digit"].get(d, 0) for d in digits]
    colors = [_ACCENT if c == max(counts) else "#E0E3F5" for c in counts]

    ax.bar(digits, counts, color=colors, edgecolor=_BORDER, linewidth=0.6)
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits], color=_TEXT, fontsize=9)
    ax.set_ylabel("Count", color=_MUTED, fontsize=8)
    ax.tick_params(axis="y", colors=_MUTED, labelsize=7)
    ax.spines[:].set_color(_BORDER)
    ax.grid(axis="y", color=_BORDER, linewidth=0.4, linestyle="--")
    ax.set_title("Samples per digit", color=_TEXT, fontsize=9, pad=6)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    records = load_csv_as_records()
    if records:
        compact = [{"#": i + 1, "user": r["user_name"], "label": r["label"]}
                   for i, r in enumerate(records)]
        df = pd.DataFrame(compact)
        st.dataframe(df, use_container_width=True, height=300)
        st.caption(
            f"Showing user + label only — the downloaded CSV also "
            f"includes pixel_0..pixel_783 ({len(records)} rows × 786 columns)."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Download dataset</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    csv_bytes = get_csv_bytes()
    if csv_bytes:
        c1.download_button("⬇ mnist_dataset.csv", csv_bytes,
                           "mnist_dataset.csv", "text/csv",
                           use_container_width=True)

    imgs_bytes, lbls_bytes = get_npy_bytes()
    if imgs_bytes:
        c2.download_button("⬇ dataset.npy", imgs_bytes,
                           "mnist_dataset.npy", "application/octet-stream",
                           use_container_width=True)
    if lbls_bytes:
        c3.download_button("⬇ labels.npy", lbls_bytes,
                           "mnist_labels.npy", "application/octet-stream",
                           use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _css()

    if "screen" not in st.session_state:
        st.session_state["screen"] = "welcome"
    if "session_index" not in st.session_state:
        st.session_state["session_index"] = 0

    screen = st.session_state.get("screen", "welcome")

    if screen == "welcome":
        _screen_welcome()
        return

    user_name = st.session_state.get("user_name", "User")
    cfg = _sidebar(user_name)

    if screen == "dataset_only":
        st.markdown(
            f"<h1 style='font-size:1.6rem;margin-bottom:0'>✏️ MNIST Canvas</h1>",
            unsafe_allow_html=True,
        )
        if st.button("← Back to drawing"):
            st.session_state["screen"] = "canvas"
            st.rerun()
        _tab_dataset()
        return

    session_index = st.session_state.get("session_index", 0)

    # ── session finished — show completion screen ──────────────────────────
    if session_index >= TOTAL_DIGITS:
        _screen_session_complete(user_name)
        return

    st.markdown(
        f"<h1 style='font-size:1.6rem;margin-bottom:0'>✏️ MNIST Canvas</h1>"
        f"<p style='color:{_MUTED};font-size:0.88rem;margin-top:0.2rem'>"
        f"Hello <b style='color:{_ACCENT}'>{user_name}</b> · "
        f"draw each digit when prompted — it's saved automatically, then "
        f"the next digit appears</p>",
        unsafe_allow_html=True,
    )

    tab_draw, tab_dataset = st.tabs(["🎨  Draw", "📊  Dataset"])

    with tab_draw:
        _render_progress(session_index)

        left, right = st.columns([1, 1.5], gap="large")

        with left:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            _render_target_digit(session_index)

            try:
                from streamlit_drawable_canvas import st_canvas
            except ImportError:
                st.error("Run: pip install streamlit-drawable-canvas")
                return

            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=cfg["stroke_width"],
                stroke_color=cfg["stroke_color"],
                background_color="#000000",
                height=cfg["canvas_size"],
                width=cfg["canvas_size"],
                drawing_mode="freedraw",
                key=f"canvas_{session_index}",
                display_toolbar=True,
            )

            c1, c2 = st.columns(2)
            process_btn = c1.button(
                f"✓ Save \"{session_index}\" & continue",
                type="primary", use_container_width=True,
            )
            clear_btn = c2.button("🗑 Clear", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if clear_btn:
                st.session_state.pop("last_result", None)
                st.rerun()

        with right:
            if process_btn:
                if canvas_result is None or canvas_result.image_data is None:
                    st.info("Draw something first.")
                elif canvas_result.image_data.max() == 0:
                    st.warning("Canvas is empty — draw the digit first.")
                else:
                    raw: np.ndarray = canvas_result.image_data
                    proc = MNISTProcessor()
                    mnist_img = proc.process(raw)

                    if mnist_img is None:
                        st.warning("No digit detected. Try a thicker or larger stroke.")
                    else:
                        # ── automatic save — label is always the digit the
                        # session is currently asking for, never ambiguous.
                        try:
                            save_sample(
                                image=mnist_img,
                                digit=int(session_index),
                                user_name=user_name,
                                confidence=0.0,
                            )
                            # Invalidate cached reads so sidebar/dataset
                            # tab reflect this new sample immediately,
                            # rather than showing stale counts for up to
                            # 15s (the cache TTL).
                            clear_sheet_cache()

                            st.session_state["last_result"] = {
                                "proc": proc, "mnist_img": mnist_img,
                                "saved_as": session_index,
                            }
                            # advance to next digit in the session
                            st.session_state["session_index"] = session_index + 1
                            st.rerun()
                        except Exception as e:
                            if "429" in str(e) or "Quota exceeded" in str(e):
                                st.error(
                                    "Google Sheets is temporarily rate-limited "
                                    "(too many requests this minute). Please "
                                    "wait about 30–60 seconds, then click Save again — "
                                    "your drawing has NOT been lost, just try again."
                                )
                            else:
                                st.error(f"Save failed: {e}")

            if "last_result" not in st.session_state:
                st.info(f"Draw the digit **{session_index}**, then click Save.")
            else:
                _render_results(user_name, cfg, st.session_state["last_result"])

    with tab_dataset:
        _tab_dataset()


if __name__ == "__main__":
    main()
