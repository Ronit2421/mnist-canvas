"""
app.py  ·  MNIST Canvas
────────────────────────
Flow:
  1. Welcome screen  → user enters their name
  2. Canvas          → draw a digit
  3. Process         → MNIST preprocessing pipeline
  4. Confirm & Save  → user confirms the digit label, auto-saved to dataset
  5. Dataset tab     → browse, download CSV / NPY

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
from dataset_manager import (
    save_sample,
    dataset_summary,
    total_samples,
    get_npy_bytes,
    get_csv_bytes,
    get_grid_csv_bytes,
    load_csv_as_records,
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

      /* ── card wrapper used around major sections ── */
      .ui-card {{
          background: {_CARD};
          border: 1px solid {_BORDER};
          border-radius: 16px;
          padding: 1.4rem 1.5rem;
          box-shadow: 0 1px 3px rgba(17,24,39,0.04), 0 1px 2px rgba(17,24,39,0.03);
          margin-bottom: 1rem;
      }}

      /* ── metric cards ── */
      [data-testid="stMetric"] {{
          background: {_SURFACE};
          border: 1px solid {_BORDER};
          border-radius: 12px;
          padding: 14px 16px;
      }}
      [data-testid="stMetricValue"] {{
          color: {_TEXT};
          font-weight: 700;
          font-size: 1.15rem;
      }}
      [data-testid="stMetricLabel"] {{ color: {_MUTED}; font-size: 0.78rem; }}

      .section-label {{
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: {_MUTED};
          font-weight: 600;
          margin-bottom: 0.5rem;
      }}

      /* ── welcome card ── */
      .welcome-wrap {{
          display: flex; align-items: center; justify-content: center;
          min-height: 70vh;
      }}
      .welcome-card {{
          background: {_CARD};
          border: 1px solid {_BORDER};
          border-radius: 24px;
          padding: 3rem 2.5rem;
          max-width: 440px;
          width: 100%;
          box-shadow: 0 20px 50px -12px rgba(99,102,241,0.18), 0 4px 12px rgba(17,24,39,0.04);
          text-align: center;
      }}
      .welcome-icon {{
          width: 56px; height: 56px; margin: 0 auto 1.2rem;
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2});
          border-radius: 16px;
          display: flex; align-items: center; justify-content: center;
          font-size: 1.6rem;
      }}
      .welcome-title {{
          font-size: 1.7rem; font-weight: 800; color: {_TEXT};
          margin-bottom: 0.3em;
      }}
      .welcome-sub {{
          font-size: 0.9rem; color: {_MUTED};
          margin-bottom: 1.8rem; line-height: 1.5;
      }}

      /* ── badges ── */
      .saved-badge {{
          background: {_GREEN}14; border: 1px solid {_GREEN}55;
          color: #047857; border-radius: 10px; padding: 0.55rem 1rem;
          font-size: 0.85rem; font-weight: 600;
          display: inline-flex; align-items: center; gap: 0.4rem;
      }}

      /* ── dataset table ── */
      [data-testid="stDataFrame"] {{
          background: {_CARD};
          border: 1px solid {_BORDER};
          border-radius: 12px;
      }}

      hr {{ border-color: {_BORDER}; margin: 1.2rem 0; }}

      /* ── buttons ── */
      [data-testid="stDownloadButton"] button,
      .stButton button {{
          border-radius: 10px;
          font-weight: 600;
          font-size: 0.86rem;
          border: 1px solid {_BORDER};
          transition: all .15s ease;
      }}
      [data-testid="stDownloadButton"] button {{
          background: {_CARD}; color: {_TEXT};
      }}
      [data-testid="stDownloadButton"] button:hover {{
          border-color: {_ACCENT}; color: {_ACCENT};
      }}
      button[kind="primary"] {{
          background: linear-gradient(135deg, {_ACCENT}, {_ACCENT2}) !important;
          border: none !important; color: white !important;
          box-shadow: 0 4px 14px rgba(99,102,241,0.3);
      }}
      button[kind="primary"]:hover {{
          opacity: 0.92;
      }}

      /* ── tabs ── */
      [data-baseweb="tab-list"] {{
          gap: 4px;
      }}
      [data-baseweb="tab"] {{
          border-radius: 10px 10px 0 0;
          font-weight: 600;
      }}

      /* ── slider / toggle accent ── */
      [data-testid="stSlider"] [role="slider"] {{
          background-color: {_ACCENT} !important;
      }}

      /* ── dark canvas, styled to sit inside a light card ── */
      .ui-card canvas {{
          border-radius: 12px;
          box-shadow: 0 0 0 1px {_BORDER};
      }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (only shown on canvas / dataset screens)
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

        if st.button("← Change name", use_container_width=True):
            for k in ["user_name", "screen", "last_result"]:
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
          Draw digits, generate MNIST-compatible images,<br>
          and contribute to a shared dataset.
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
        if st.button("Start drawing  →", type="primary", use_container_width=True):
            name = name.strip()
            if not name:
                st.error("Please enter your name to continue.")
            elif len(name) < 2:
                st.error("Name must be at least 2 characters.")
            else:
                st.session_state["user_name"] = name
                st.session_state["screen"] = "canvas"
                st.rerun()

        st.markdown(
            f"<div style='text-align:center;color:{_MUTED};font-size:0.78rem;"
            f"margin-top:1rem'>{total_samples()} drawings saved so far</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Canvas + Results rendering helper (called from main())
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(user_name: str, cfg: dict, result: dict) -> None:
    """Render the right-panel results from a stored result dict."""
    proc      = result["proc"]
    mnist_img = result["mnist_img"]

    stats     = proc.get_stats()
    png_bytes = proc.to_png_bytes()
    npy_bytes = proc.to_numpy_bytes()

    # ── 28×28 image + download ───────────────────────────────────────────────
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    img_col, dl_col = st.columns([1, 1], gap="medium")

    with img_col:
        st.markdown("<div class='section-label'>MNIST output</div>",
                    unsafe_allow_html=True)
        render_mnist_image(mnist_img)

    with dl_col:
        st.markdown("<div class='section-label'>Download image</div>",
                    unsafe_allow_html=True)
        if png_bytes and npy_bytes:
            render_download_buttons(png_bytes, npy_bytes)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Image stats</div>",
                    unsafe_allow_html=True)
        render_stats(stats)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── dataset status / label confirmation ──────────────────────────────────
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Dataset</div>", unsafe_allow_html=True)

    save_col, label_col = st.columns([1, 1], gap="medium")

    saved_as    = result.get("saved_as", 0)
    saved_fname = result.get("saved_fname")

    with save_col:
        st.markdown(
            f"<div class='saved-badge'>✓ Saved as digit {saved_as}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Saved as a flattened CSV row AND as a real 28×28 grid file "
            "you can open and see the digit's shape in."
        )
        if saved_fname:
            grid_bytes = get_grid_csv_bytes(saved_fname)
            if grid_bytes:
                grid_csv_name = f"digit_{saved_as}_{saved_fname}.csv"
                st.download_button(
                    "⬇ Download 28×28 grid CSV",
                    grid_bytes, grid_csv_name, "text/csv",
                    use_container_width=True,
                )

    with label_col:
        confirmed_digit = st.selectbox(
            "Not correct? Fix the label",
            options=list(range(10)),
            index=int(saved_as),
            key="confirm_digit",
        )
        if confirmed_digit != saved_as:
            if st.button("↻ Re-save with corrected label",
                         type="primary", use_container_width=True):
                fname = save_sample(
                    image=mnist_img,
                    digit=int(confirmed_digit),
                    user_name=user_name,
                    confidence=0.0,
                )
                result["saved_as"]    = confirmed_digit
                result["saved_fname"] = fname
                st.session_state["last_result"] = result
                st.success(f"Re-saved as digit {confirmed_digit}")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── pipeline steps ───────────────────────────────────────────────────────
    if cfg["show_steps"]:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Preprocessing pipeline</div>",
                    unsafe_allow_html=True)
        render_pipeline_steps(proc)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── pixel matrix ─────────────────────────────────────────────────────────
    if cfg["show_matrix"]:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Pixel matrix (28×28 · uint8)</div>",
                    unsafe_allow_html=True)
        render_pixel_matrix(mnist_img)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tab
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
    m1, m2, m3 = st.columns(3)
    m1.metric("Total samples", n)
    m2.metric("Contributors", len(summary["contributors"]))
    most_drawn = max(summary["per_digit"], key=summary["per_digit"].get)
    m3.metric("Most drawn digit", f"{most_drawn}  ({summary['per_digit'][most_drawn]}×)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── per-digit bar chart ──────────────────────────────────────────────────
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

    # ── records table ────────────────────────────────────────────────────────
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

    # ── downloads ─────────────────────────────────────────────────────────────
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Download dataset</div>",
                unsafe_allow_html=True)

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

    screen = st.session_state.get("screen", "welcome")

    if screen == "welcome":
        _screen_welcome()
        return

    user_name = st.session_state.get("user_name", "User")
    cfg = _sidebar(user_name)

    st.markdown(
        f"<h1 style='font-size:1.6rem;margin-bottom:0'>✏️ MNIST Canvas</h1>"
        f"<p style='color:{_MUTED};font-size:0.88rem;margin-top:0.2rem'>"
        f"Hello <b style='color:{_ACCENT}'>{user_name}</b> · "
        f"draw a digit and click Process — it's saved to the dataset automatically</p>",
        unsafe_allow_html=True,
    )

    tab_draw, tab_dataset = st.tabs(["🎨  Draw", "📊  Dataset"])

    with tab_draw:
        left, right = st.columns([1, 1.5], gap="large")

        with left:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Draw here</div>",
                        unsafe_allow_html=True)
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
                key="canvas",
                display_toolbar=True,
            )

            c1, c2 = st.columns(2)
            process_btn = c1.button("▶ Process", type="primary",
                                    use_container_width=True)
            clear_btn   = c2.button("🗑 Clear",  use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if clear_btn:
                st.session_state.pop("last_result", None)
                st.rerun()

        with right:
            if process_btn:
                if canvas_result is None or canvas_result.image_data is None:
                    st.info("Draw something first.")
                elif canvas_result.image_data.max() == 0:
                    st.warning("Canvas is empty — draw a digit first.")
                else:
                    raw: np.ndarray = canvas_result.image_data
                    proc = MNISTProcessor()
                    mnist_img = proc.process(raw)

                    if mnist_img is None:
                        st.warning("No digit detected. Try a thicker or larger stroke.")
                    else:
                        # ── default label: 0, user confirms/corrects below ──
                        auto_label = 0
                        auto_fname = save_sample(
                            image=mnist_img,
                            digit=auto_label,
                            user_name=user_name,
                            confidence=0.0,
                        )

                        st.session_state["last_result"] = {
                            "proc":        proc,
                            "mnist_img":   mnist_img,
                            "saved_as":    auto_label,
                            "saved_fname": auto_fname,
                        }

            if "last_result" not in st.session_state:
                st.info("Draw a digit on the canvas, then click **▶ Process**.")
            else:
                _render_results(user_name, cfg, st.session_state["last_result"])

    with tab_dataset:
        _tab_dataset()


if __name__ == "__main__":
    main()
