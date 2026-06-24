# ✏️ MNIST Canvas

A Streamlit app where anyone can draw a handwritten digit, get it converted
into a true **MNIST-compatible 28×28 image**, see a live digit prediction,
and contribute it to a **shared, persistent dataset** — visible to every
user of the deployed app, not just stored on one person's machine.

---

## How it works

1. Enter your name (just for tagging contributions)
2. Draw a digit on the canvas
3. Click **▶ Process** →
   - Runs the real MNIST preprocessing pipeline (grayscale → denoise →
     crop → resize → centre-of-mass alignment, exactly like the original
     1998 MNIST dataset)
   - A scikit-learn classifier predicts the digit + shows confidence
   - The drawing is **auto-saved** to the shared dataset
4. Wrong prediction? Pick the correct digit and re-save with the fixed label
5. **Dataset tab** — see totals, per-digit counts, contributors, and
   download the whole dataset as CSV or NumPy arrays

---

## Two storage modes

| Mode | When | Where data goes |
|------|------|------------------|
| **Local fallback** | No Supabase configured | `dataset/` folder on disk — single machine only, wiped on restart if deployed |
| **Cloud (Supabase)** | `SUPABASE_URL` + `SUPABASE_KEY` set | Shared Postgres table — every user's drawings land in the same place, persists forever |

For a **live, multi-user app**, you need Cloud mode. Setup below.

---

## Quick Start (local testing, no cloud setup needed)

```bash
cd mnist_app
pip install -r requirements.txt
streamlit run app.py
```

Without Supabase configured, it still works — drawings save to a local
`dataset/` folder. Good for trying things out before deploying.

---

## Deploying live with a shared dataset (Supabase + Streamlit Cloud)

### 1. Create a free Supabase project

1. Go to [supabase.com](https://supabase.com) → sign up → **New project**
2. Wait ~2 minutes for it to provision

### 2. Create the table

1. In your Supabase project → **SQL Editor** → **New query**
2. Open `supabase_setup.sql` from this folder, paste its contents, click **Run**

This creates the `mnist_samples` table with the right columns, indexes,
and public insert/read policies.

### 3. Get your API credentials

1. Project → **Settings** → **API**
2. Copy the **Project URL** and the **anon public key**

### 4. Push this project to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mnist-canvas.git
git push -u origin main
```

> `.gitignore` already excludes `dataset/`, `*.joblib`, and
> `.streamlit/secrets.toml` — your local data and secrets never get
> pushed.

### 5. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. **New app** → pick your repo → branch `main` → main file `app.py`
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:

```toml
SUPABASE_URL = "https://YOUR-PROJECT-REF.supabase.co"
SUPABASE_KEY = "YOUR-ANON-PUBLIC-KEY"
```

4. Click **Deploy**

Your app is now live at `https://your-app-name.streamlit.app`, and every
visitor's drawings save to the same shared Supabase table.

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

## Classifier

`RandomForestClassifier(n_estimators=500)` trained on scikit-learn's
built-in `load_digits` dataset (upscaled 8×8 → 28×28). No internet
download required, trains in a few seconds on first launch, then cached
to `mnist_clf.joblib` (regenerated automatically — also git-ignored).

---

## Dataset formats

Every saved drawing is stored two ways:

1. **Flattened** (ML-ready, Kaggle-style): `user, label, pixel_0, ..., pixel_783`
   — one row per sample, downloadable as `mnist_dataset.csv`
2. **Grid** (human-readable): a real 28×28 grid you can open and visually
   see the digit's shape in — downloadable per-sample from the result panel

Plus `.npy` exports (`dataset.npy` shape `(N,28,28)`, `labels.npy` shape `(N,)`)
for quick numpy-based model training.
