"""
ui_components.py
────────────────
Reusable Streamlit UI helpers — pixel-perfect MNIST image display.
Modern light theme: clean cards, soft shadows, Inter typeface.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from PIL import Image

# ── modern light palette (mirrors app.py) ───────────────────────────────────
_BG       = "#FFFFFF"
_SURFACE  = "#F7F8FA"
_ACCENT   = "#6366F1"   # indigo
_ACCENT2  = "#EC4899"   # pink
_TEXT     = "#111827"
_MUTED    = "#6B7280"
_GRID     = "#E5E7EB"
_GREEN    = "#10B981"


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL-PERFECT 28×28 display
# Pure PIL nearest-neighbour upscale — guarantees an EXACT integer multiple
# of 28×28 with zero blur. Every one of the 784 source pixels maps to a
# crisp square in the output.
# ─────────────────────────────────────────────────────────────────────────────

def render_mnist_image(img: np.ndarray, cell_px: int = 16) -> None:
    """
    Render the 28×28 MNIST image at high resolution with a subtle grid,
    using exact nearest-neighbour pixel blocks (no blur, no smoothing).
    """
    H, W = img.shape  # 28, 28

    big = Image.fromarray(img, mode="L").resize(
        (W * cell_px, H * cell_px), resample=Image.NEAREST
    ).convert("RGB")

    arr = np.array(big)
    grid_color = (235, 235, 240)  # faint light-gray grid, visible on black bg
    for i in range(0, H + 1):
        y = min(i * cell_px, arr.shape[0] - 1)
        arr[y, :, :] = np.where(
            arr[y, :, :].sum(axis=-1, keepdims=True) < 40,
            grid_color, arr[y, :, :]
        )
    for j in range(0, W + 1):
        x = min(j * cell_px, arr.shape[1] - 1)
        arr[:, x, :] = np.where(
            arr[:, x, :].sum(axis=-1, keepdims=True) < 40,
            grid_color, arr[:, x, :]
        )

    final = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)

    st.image(final, width=min(W * cell_px, 448))


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-matrix heatmap (real HTML table — always crisp)
# ─────────────────────────────────────────────────────────────────────────────

def render_pixel_matrix(img: np.ndarray) -> None:
    """Show the 28×28 pixel values as a real HTML table."""
    H, W = img.shape
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_indigo", ["#FFFFFF", "#6366F1"], N=256
    )

    rows_html = []
    for r in range(H):
        cells = []
        for c in range(W):
            val = int(img[r, c])
            rgba = cmap(val / 255.0)
            r8, g8, b8 = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            text_col = "#FFFFFF" if val > 140 else "#9CA3AF"
            cells.append(
                f'<td style="background:rgb({r8},{g8},{b8});color:{text_col};'
                f'font-family:monospace;font-size:9px;text-align:center;'
                f'padding:0;width:18px;height:18px;border:1px solid #EEF0F4;">'
                f'{val}</td>'
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <div style="overflow:auto;max-width:100%;border:1px solid {_GRID};
                border-radius:12px;background:{_SURFACE};padding:8px;">
      <table style="border-collapse:collapse;margin:0 auto;">
        {''.join(rows_html)}
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("Pixel matrix — 28×28 uint8 values (0 = black, 255 = white)")


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def render_stats(stats: dict) -> None:
    cols = st.columns(3)
    metrics = [
        ("Shape",    f"{stats.get('shape', '—')}"),
        ("dtype",    stats.get("dtype", "—")),
        ("Min px",   stats.get("min", "—")),
        ("Max px",   stats.get("max", "—")),
        ("Mean px",  stats.get("mean", "—")),
        ("Non-zero", f"{stats.get('nonzero_pixels', '—')} px"),
    ]
    for i, (label, value) in enumerate(metrics):
        cols[i % 3].metric(label, value)


# ─────────────────────────────────────────────────────────────────────────────
# Download buttons
# ─────────────────────────────────────────────────────────────────────────────

def render_download_buttons(png_bytes: bytes, npy_bytes: bytes) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("⬇ PNG",  png_bytes,
                       "mnist_digit.png", "image/png",
                       use_container_width=True)
    c2.download_button("⬇ .npy", npy_bytes,
                       "mnist_digit.npy", "application/octet-stream",
                       use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step visualiser
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_steps(processor) -> None:
    steps = [
        ("Gray",              processor.step_grayscale),
        ("Denoised",          processor.step_denoised),
        ("Binary (Otsu)",     processor.step_binary),
        ("Closed (gaps)",     processor.step_closed),
        ("Cropped",           processor.step_cropped),
        ("Resized (20px)",    processor.step_resized),
    ]
    valid = [(t, im) for t, im in steps if im is not None]
    n = len(valid)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.4))
    fig.patch.set_facecolor(_BG)

    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, valid):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, color=_TEXT, fontsize=8, pad=6,
                     fontfamily="sans-serif")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)
