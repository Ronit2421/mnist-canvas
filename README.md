# ✏️ MNIST Canvas

A Streamlit app that guides each visitor through drawing **every digit
from 0 to 9, in order**, converts each into a true **MNIST-compatible
28×28 image**, and automatically saves it to a **shared Google Sheet** —
crowdsourcing a real, growing, correctly-labelled handwritten digit
dataset.

---

## How it works

1. Enter your name
2. You're shown digit **0** — draw it, click Save → it's saved
   automatically (the label is never ambiguous, since the app is always
   asking for one specific digit)
3. The app advances to digit **1**, then **2**, ... up to **9**
4. After digit 9 is saved → "Session complete!" screen, with the option
   to start a brand new 0–9 session
5. **Dataset tab** — totals, per-digit counts, contributors, and
   download buttons (CSV / NumPy) — reads live from the shared sheet

---

## Two storage modes

| Mode | When | Where data goes |
|------|------|------------------|
| **Local fallback** | No Google credentials configured | `dataset/` folder on disk — single machine only |
| **Cloud (Google Sheets)** | `gcp_service_account` + `GOOGLE_SHEET_ID` set in secrets | Shared Google Sheet — every visitor's drawings land in the same place, persists forever |

For a **live, multi-user app**, you need Cloud mode. Setup below.

---

## Quick Start (local testing, no cloud setup needed)

```bash
cd mnist_app
pip install -r requirements.txt
streamlit run app.py
```

Without Google Sheets configured, it still works — drawings save to a
local `dataset/` folder. Good for trying things out before deploying.

---

## Deploying live with a shared Google Sheets dataset

### 1. Create a Google Cloud service account

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create or select a project
2. **APIs & Services → Library** → enable **Google Sheets API** and **Google Drive API**
3. **IAM & Admin → Service Accounts** → **Create Service Account**
4. Once created, open it → **Keys** tab → **Add Key → Create new key → JSON**
5. This downloads a `.json` file — keep it safe, you'll copy values from it next

### 2. Create the Google Sheet

1. Go to [sheets.google.com](https://sheets.google.com) → create a new blank sheet
2. Name it anything (e.g. "MNIST Dataset")
3. Click **Share** → paste the service account's `client_email` (found in
   the downloaded JSON) → give it **Editor** access
4. Copy the **Sheet ID** from the URL:
   ```
   https://docs.google.com/spreadsheets/d/THIS_PART_IS_THE_ID/edit
   ```

### 3. Fill in your secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
fill in every field using values from your downloaded JSON key, plus the
Sheet ID from step 2.

### 4. Push this project to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mnist-canvas.git
git push -u origin main
```

> `.gitignore` already excludes `dataset/` and `.streamlit/secrets.toml`
> — your local data and secrets never get pushed.

### 5. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. **New app** → pick your repo → branch `main` → main file `app.py`
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste
   the entire contents of your filled-in `secrets.toml`
4. Click **Deploy**

Your app is now live, and every visitor's drawings save to the same
shared Google Sheet.

---

## MNIST Preprocessing Pipeline

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Grayscale | RGBA → L via alpha channel |
| 2 | Bilateral denoise | Edge-preserving smoothing |
| 3 | Otsu threshold | Auto adaptive binarisation (used to locate the digit) |
| 4 | Morphological closing | Fills small gaps in strokes |
| 5 | Bounding-box crop | + padding guard |
| 6 | Aspect-preserving resize | Longest side = 20 px |
| 7 | 28×28 frame pad | Geometric centre paste |
| 8 | Centre-of-mass alignment | `scipy.ndimage.shift` → CoM at (14,14) |

Anti-aliased grayscale edges are preserved — this is what makes the
zoomed-in pixel grid look like authentic soft-edged MNIST digits instead
of harsh jagged silhouettes.

Output: `(28, 28)` `uint8`, values `[0, 255]`, white digit on black background.

---

## Dataset format

Google Sheet columns: `timestamp, user_name, label, pixel_0, ..., pixel_783`
— one row per drawing.

Downloadable as:
- `mnist_dataset.csv` — flattened, ML-ready (`user, label, pixel_0..pixel_783`)
- `dataset.npy` / `labels.npy` — numpy arrays, shapes `(N,28,28)` and `(N,)`
- Per-drawing 28×28 grid CSV — open it and visually see the digit's shape
