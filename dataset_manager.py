"""
dataset_manager.py
──────────────────
Shared, persistent dataset storage using Supabase (Postgres).

Every user's drawing is saved to a single shared cloud table, so the
dataset survives app restarts/sleeps and is visible to ALL users of the
live app — not just stored on one person's local disk.

Table schema (created once via the SQL in supabase_setup.sql):

    mnist_samples
    ──────────────
    id          bigint        primary key, auto-increment
    user_name   text          who drew it
    label       smallint      confirmed digit 0–9
    pixels      jsonb         784 ints, row-major 28×28, values 0–255
    confidence  real          classifier confidence at save time (0–1)
    created_at  timestamptz   default now()

Connection is configured via Streamlit secrets (.streamlit/secrets.toml
locally, or the "Secrets" panel on Streamlit Community Cloud):

    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "your-anon-public-key"

If secrets are missing (e.g. running with no internet/Supabase set up
yet), this module transparently falls back to local CSV/NPY files in
./dataset/ so the app still works for local testing.
"""

from __future__ import annotations

import os
import csv
import json
import numpy as np
from datetime import datetime
from PIL import Image

_TABLE = "mnist_samples"

# ── local fallback paths (used only if Supabase isn't configured) ──────────
_DS_ROOT  = os.path.join(os.path.dirname(__file__), "dataset")
_CSV_PATH = os.path.join(_DS_ROOT, "mnist_dataset.csv")
_PIXEL_COLS = [f"pixel_{i}" for i in range(784)]
_CSV_HEADER = ["user", "label"] + _PIXEL_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Supabase client (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_client = None
_supabase_checked = False
_supabase_available = False


def _get_client():
    """Return a cached Supabase client, or None if not configured."""
    global _client, _supabase_checked, _supabase_available

    if _supabase_checked:
        return _client

    _supabase_checked = True
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except Exception:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        _supabase_available = False
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        _supabase_available = True
    except Exception:
        _client = None
        _supabase_available = False

    return _client


def is_cloud_mode() -> bool:
    """True if Supabase is configured and reachable."""
    _get_client()
    return _supabase_available


# ─────────────────────────────────────────────────────────────────────────────
# Public API — save_sample
# ─────────────────────────────────────────────────────────────────────────────

def save_sample(
    image: np.ndarray,
    digit: int,
    user_name: str,
    confidence: float = 0.0,
) -> str:
    """
    Persist one 28×28 drawing.

    If Supabase is configured: inserts one row into the shared cloud
    table `mnist_samples` (visible to every user of the live app).

    If not configured: falls back to a local CSV file (single-machine
    only — for local testing without Supabase).

    Parameters
    ----------
    image       : np.ndarray  shape (28, 28) uint8
    digit       : int         0–9 confirmed label
    user_name   : str
    confidence  : float       0–1 classifier confidence

    Returns
    -------
    identifier : str   row id (cloud mode) or filename stem (local mode)
    """
    flat_pixels = image.astype(np.uint8).flatten().tolist()  # 784 ints, row-major

    client = _get_client()
    if client is not None:
        resp = client.table(_TABLE).insert({
            "user_name":  user_name,
            "label":      int(digit),
            "pixels":     flat_pixels,
            "confidence": float(confidence),
        }).execute()
        row_id = resp.data[0]["id"] if resp.data else "unknown"
        return str(row_id)

    # ── local fallback ───────────────────────────────────────────────────────
    return _save_local(image, digit, user_name, flat_pixels)


def _save_local(image: np.ndarray, digit: int, user_name: str,
                flat_pixels: list[int]) -> str:
    os.makedirs(_DS_ROOT, exist_ok=True)
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(c for c in user_name if c.isalnum() or c in "_-") or "user"
    stem = f"{safe_name}_{digit}_{ts}"

    with open(_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([user_name, int(digit)] + flat_pixels)

    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Public API — read helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_rows() -> list[dict]:
    """Fetch every row from Supabase (or local CSV fallback)."""
    client = _get_client()
    if client is not None:
        resp = client.table(_TABLE).select(
            "id, user_name, label, confidence, created_at"
        ).order("id", desc=False).execute()
        return resp.data or []

    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return [{"user_name": r["user"], "label": int(r["label"])} for r in reader]


def total_samples() -> int:
    """Number of samples saved so far (cloud or local)."""
    client = _get_client()
    if client is not None:
        resp = client.table(_TABLE).select("id", count="exact").execute()
        return resp.count or 0
    return len(_fetch_all_rows())


def dataset_summary() -> dict:
    """Return per-digit counts and contributor list."""
    rows = _fetch_all_rows()
    if not rows:
        return {"total": 0, "per_digit": {}, "contributors": []}

    from collections import Counter
    digit_counts = Counter(int(r["label"]) for r in rows)
    contributors  = sorted(set(r["user_name"] for r in rows))

    return {
        "total":        len(rows),
        "per_digit":    {d: digit_counts.get(d, 0) for d in range(10)},
        "contributors": contributors,
    }


def load_csv_as_records() -> list[dict]:
    """Return all rows as user/label dicts (for the dataset table view)."""
    return _fetch_all_rows()


# ─────────────────────────────────────────────────────────────────────────────
# Public API — bulk export (download buttons)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_pixels() -> list[dict]:
    """Fetch full rows INCLUDING pixel arrays — used only for export."""
    client = _get_client()
    if client is not None:
        resp = client.table(_TABLE).select(
            "id, user_name, label, pixels, confidence, created_at"
        ).order("id", desc=False).execute()
        return resp.data or []

    if not os.path.exists(_CSV_PATH):
        return []
    out = []
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pixels = [int(r[f"pixel_{i}"]) for i in range(784)]
            out.append({"user_name": r["user"], "label": int(r["label"]),
                       "pixels": pixels})
    return out


def get_csv_bytes() -> bytes | None:
    """Build mnist_dataset.csv on the fly: user,label,pixel_0..pixel_783."""
    rows = _fetch_all_pixels()
    if not rows:
        return None

    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_HEADER)
    for r in rows:
        writer.writerow([r["user_name"], r["label"]] + list(r["pixels"]))
    return buf.getvalue().encode()


def get_npy_bytes() -> tuple[bytes, bytes] | tuple[None, None]:
    """Build dataset.npy + labels.npy on the fly from all cloud/local rows."""
    rows = _fetch_all_pixels()
    if not rows:
        return None, None

    import io
    imgs = np.array([r["pixels"] for r in rows], dtype=np.uint8).reshape(-1, 28, 28)
    lbls = np.array([r["label"] for r in rows], dtype=np.int32)

    imgs_buf, lbls_buf = io.BytesIO(), io.BytesIO()
    np.save(imgs_buf, imgs)
    np.save(lbls_buf, lbls)
    return imgs_buf.getvalue(), lbls_buf.getvalue()


def get_grid_csv_bytes(identifier: str) -> bytes | None:
    """
    Build a single 28×28 grid CSV (human-readable) for one saved sample,
    identified by its row id (cloud mode) or filename stem (local mode).
    """
    client = _get_client()
    if client is not None:
        try:
            resp = client.table(_TABLE).select("pixels").eq(
                "id", int(identifier)
            ).single().execute()
            pixels = resp.data["pixels"]
        except Exception:
            return None
    else:
        # Local fallback: identifier is a CSV row position not tracked
        # individually — rebuild from the most recent local save instead.
        rows = _fetch_all_pixels()
        if not rows:
            return None
        pixels = rows[-1]["pixels"]

    arr = np.array(pixels, dtype=np.uint8).reshape(28, 28)
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in arr:
        writer.writerow(row.tolist())
    return buf.getvalue().encode()
