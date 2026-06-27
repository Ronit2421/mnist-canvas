"""
sheets_manager.py
──────────────────
Shared, persistent dataset storage using Google Sheets.

Every user's drawing is appended as a row to a single shared Google
Sheet, so the dataset survives app restarts and is visible to ALL users
of the live app — not just stored on one person's local disk.

Sheet layout (row 1 = header, auto-created on first run):

    timestamp | user_name | label | pixel_0 | pixel_1 | ... | pixel_783

Authentication uses a Google Cloud **service account** (not OAuth login —
no user-facing Google sign-in popup). Configure via Streamlit secrets
(.streamlit/secrets.toml locally, or the "Secrets" panel on Streamlit
Community Cloud):

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "...@....iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"

    GOOGLE_SHEET_ID = "the-id-from-the-sheet-url"

The target sheet must be shared (Editor access) with the
`client_email` above, or every write will fail with a permission error.

If secrets are missing (e.g. running locally with no Google Cloud setup
yet), this module transparently falls back to local CSV/NPY files in
./dataset/ so the app still works for local testing/dev.
"""

from __future__ import annotations

import os
import csv
import numpy as np
from datetime import datetime
from PIL import Image

# ── local fallback paths ─────────────────────────────────────────────────────
_DS_ROOT  = os.path.join(os.path.dirname(__file__), "dataset")
_IMG_DIR  = os.path.join(_DS_ROOT, "images")
_GRID_DIR = os.path.join(_DS_ROOT, "grids")
_CSV_PATH = os.path.join(_DS_ROOT, "mnist_dataset.csv")
_NPY_IMG  = os.path.join(_DS_ROOT, "dataset.npy")
_NPY_LBL  = os.path.join(_DS_ROOT, "labels.npy")

_PIXEL_COLS = [f"pixel_{i}" for i in range(784)]
_SHEET_HEADER = ["timestamp", "user_name", "label"] + _PIXEL_COLS
_LOCAL_CSV_HEADER = ["user_name", "label"] + _PIXEL_COLS

_WORKSHEET_NAME = "mnist_samples"


# ─────────────────────────────────────────────────────────────────────────────
# Cloud (Google Sheets) client — lazily constructed, cached for the process
# ─────────────────────────────────────────────────────────────────────────────

_sheet_client_cache: dict = {"worksheet": None, "checked": False}


def is_cloud_mode() -> bool:
    """True if Google Sheets credentials are present in Streamlit secrets."""
    try:
        import streamlit as st
        has_creds = "gcp_service_account" in st.secrets
        has_sheet_id = bool(st.secrets.get("GOOGLE_SHEET_ID", ""))
        return has_creds and has_sheet_id
    except Exception:
        return False


def _get_worksheet():
    """Return a cached gspread Worksheet, creating the header row if needed."""
    if _sheet_client_cache["checked"]:
        return _sheet_client_cache["worksheet"]

    _sheet_client_cache["checked"] = True

    if not is_cloud_mode():
        _sheet_client_cache["worksheet"] = None
        return None

    import streamlit as st
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)

    sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    spreadsheet = client.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet(_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=_WORKSHEET_NAME, rows=1000, cols=len(_SHEET_HEADER)
        )
        worksheet.append_row(_SHEET_HEADER, value_input_option="RAW")

    # Ensure header exists even if the worksheet already existed but was empty
    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.append_row(_SHEET_HEADER, value_input_option="RAW")

    _sheet_client_cache["worksheet"] = worksheet
    return worksheet


# ─────────────────────────────────────────────────────────────────────────────
# Local fallback helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_local() -> None:
    os.makedirs(_DS_ROOT, exist_ok=True)
    os.makedirs(_IMG_DIR, exist_ok=True)
    os.makedirs(_GRID_DIR, exist_ok=True)
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_LOCAL_CSV_HEADER)


def _save_local(image: np.ndarray, digit: int, user_name: str) -> str:
    _init_local()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(c for c in user_name if c.isalnum() or c in "_-") or "user"
    stem      = f"{safe_name}_{digit}_{ts}"

    Image.fromarray(image, mode="L").save(os.path.join(_IMG_DIR, f"{stem}.png"))

    with open(os.path.join(_GRID_DIR, f"{stem}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for row in image.astype(np.uint8):
            writer.writerow(row.tolist())

    flat = image.astype(np.uint8).flatten().tolist()
    with open(_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([user_name, int(digit)] + flat)

    if os.path.exists(_NPY_IMG):
        imgs = np.load(_NPY_IMG)
        lbls = np.load(_NPY_LBL)
    else:
        imgs = np.empty((0, 28, 28), dtype=np.uint8)
        lbls = np.empty((0,), dtype=np.int32)
    imgs = np.concatenate([imgs, image[np.newaxis]], axis=0)
    lbls = np.concatenate([lbls, np.array([digit], dtype=np.int32)], axis=0)
    np.save(_NPY_IMG, imgs)
    np.save(_NPY_LBL, lbls)

    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Public API (same shape as the old dataset_manager.py, for easy app.py reuse)
# ─────────────────────────────────────────────────────────────────────────────

def save_sample(
    image: np.ndarray,
    digit: int,
    user_name: str,
    confidence: float = 0.0,
) -> str:
    """
    Persist one 28×28 drawing.

    Cloud mode  → appends one row to the shared Google Sheet:
                  timestamp, user_name, label, pixel_0..pixel_783
    Local mode  → writes PNG + grid-CSV + appends to local mnist_dataset.csv
                  + updates dataset.npy / labels.npy

    Returns an identifier string (used later to fetch the matching
    28×28 grid CSV for download).
    """
    worksheet = _get_worksheet()

    if worksheet is not None:
        flat = image.astype(np.uint8).flatten().tolist()
        row = [datetime.now().isoformat(timespec="seconds"), user_name, int(digit)] + flat
        worksheet.append_row(row, value_input_option="RAW")
        return f"{user_name}_{digit}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    return _save_local(image, digit, user_name)


def load_csv_as_records() -> list[dict]:
    """Return all rows as a list of {'user_name':..., 'label':...} dicts."""
    worksheet = _get_worksheet()

    if worksheet is not None:
        all_values = worksheet.get_all_values()
        if len(all_values) < 2:
            return []
        header = all_values[0]
        try:
            user_idx  = header.index("user_name")
            label_idx = header.index("label")
        except ValueError:
            return []
        return [
            {"user_name": row[user_idx], "label": row[label_idx]}
            for row in all_values[1:] if len(row) > max(user_idx, label_idx)
        ]

    _init_local()
    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, newline="") as f:
        records = []
        for r in csv.DictReader(f):
            if "user_name" in r and "label" in r:
                records.append({"user_name": r["user_name"], "label": r["label"]})
        return records


def total_samples() -> int:
    """Number of samples saved so far."""
    worksheet = _get_worksheet()
    if worksheet is not None:
        try:
            return max(len(worksheet.get_all_values()) - 1, 0)
        except Exception:
            return 0
    if os.path.exists(_NPY_LBL):
        return int(np.load(_NPY_LBL).shape[0])
    return 0


def dataset_summary() -> dict:
    """Return per-digit counts and contributor list."""
    records = load_csv_as_records()
    if not records:
        return {"total": 0, "per_digit": {}, "contributors": []}

    from collections import Counter
    digit_counts = Counter()
    contributors = set()
    for r in records:
        try:
            digit_counts[int(r["label"])] += 1
        except (ValueError, TypeError):
            continue
        contributors.add(r["user_name"])

    return {
        "total":        len(records),
        "per_digit":    {d: digit_counts.get(d, 0) for d in range(10)},
        "contributors": sorted(contributors),
    }


def get_csv_bytes() -> bytes | None:
    """Return the full dataset as flattened CSV bytes (user,label,pixel_0..pixel_783)."""
    worksheet = _get_worksheet()

    if worksheet is not None:
        all_values = worksheet.get_all_values()
        if len(all_values) < 2:
            return None
        header = all_values[0]
        try:
            user_idx  = header.index("user_name")
            label_idx = header.index("label")
            pixel_idx_start = header.index("pixel_0")
        except ValueError:
            return None

        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["user", "label"] + _PIXEL_COLS)
        for row in all_values[1:]:
            if len(row) <= pixel_idx_start:
                continue
            writer.writerow(
                [row[user_idx], row[label_idx]] + row[pixel_idx_start:pixel_idx_start + 784]
            )
        return buf.getvalue().encode("utf-8")

    if not os.path.exists(_CSV_PATH):
        return None
    with open(_CSV_PATH, "rb") as f:
        return f.read()


def get_npy_bytes() -> tuple[bytes, bytes] | tuple[None, None]:
    """Return (images_npy_bytes, labels_npy_bytes) for download."""
    import io
    worksheet = _get_worksheet()

    if worksheet is not None:
        all_values = worksheet.get_all_values()
        if len(all_values) < 2:
            return None, None
        header = all_values[0]
        try:
            label_idx = header.index("label")
            pixel_idx_start = header.index("pixel_0")
        except ValueError:
            return None, None

        imgs, lbls = [], []
        for row in all_values[1:]:
            if len(row) <= pixel_idx_start:
                continue
            try:
                pixels = [int(v) for v in row[pixel_idx_start:pixel_idx_start + 784]]
                if len(pixels) != 784:
                    continue
                imgs.append(np.array(pixels, dtype=np.uint8).reshape(28, 28))
                lbls.append(int(row[label_idx]))
            except (ValueError, IndexError):
                continue

        if not imgs:
            return None, None

        imgs_arr = np.stack(imgs)
        lbls_arr = np.array(lbls, dtype=np.int32)
        imgs_buf, lbls_buf = io.BytesIO(), io.BytesIO()
        np.save(imgs_buf, imgs_arr)
        np.save(lbls_buf, lbls_arr)
        return imgs_buf.getvalue(), lbls_buf.getvalue()

    if not (os.path.exists(_NPY_IMG) and os.path.exists(_NPY_LBL)):
        return None, None
    imgs_buf, lbls_buf = io.BytesIO(), io.BytesIO()
    np.save(imgs_buf, np.load(_NPY_IMG))
    np.save(lbls_buf, np.load(_NPY_LBL))
    return imgs_buf.getvalue(), lbls_buf.getvalue()


def get_grid_csv_bytes(identifier: str) -> bytes | None:
    """
    Return a 28×28 grid CSV (human-readable layout) for one sample.

    Cloud mode: Sheets doesn't index by filename, so this always returns
    None — app.py should call image_to_grid_csv_bytes() directly on the
    in-memory image instead. Local mode: looks up dataset/grids/<id>.csv.
    """
    if is_cloud_mode():
        return None

    grid_path = os.path.join(_GRID_DIR, f"{identifier}.csv")
    if not os.path.exists(grid_path):
        return None
    with open(grid_path, "rb") as f:
        return f.read()


def image_to_grid_csv_bytes(image: np.ndarray) -> bytes:
    """Convert any 28×28 array directly to grid-CSV bytes (works in both modes)."""
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in image.astype(np.uint8):
        writer.writerow(row.tolist())
    return buf.getvalue().encode("utf-8")