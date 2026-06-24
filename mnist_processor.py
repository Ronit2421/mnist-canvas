"""
mnist_processor.py
──────────────────
MNIST-compatible preprocessing — produces authentic soft-pixel MNIST look.

Pipeline:
  1. Grayscale (alpha-aware)
  2. Mild denoise with bilateral filter (preserves edges, unlike Gaussian)
  3. Otsu threshold → clean binary (used only to find/crop the digit)
  4. Morphological closing → fill tiny gaps in strokes
  5. Bounding-box crop + padding
  6. Aspect-ratio-preserving resize, longest side = 20 px
     — downscale: INTER_AREA   (best for shrinking, gives soft AA edges)
     — upscale  : INTER_CUBIC  (smooth, no blockiness)
  7. Pad to 28×28, digit centred by centre-of-mass (LeCun 1998)
  8. Final contrast stretch → ensure max pixel = 255 (full white)

Grayscale anti-aliasing is preserved through resize + centering — this
is exactly what makes real MNIST digits look like soft pixel blocks
(visible squares with smooth gray edges) instead of harsh binary jaggies.

Output: (28, 28) uint8, values [0, 255], white digit on black background.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image


# ─────────────────────────── helpers ────────────────────────────────────────

def _to_grayscale(rgba: np.ndarray) -> np.ndarray:
    """RGBA canvas → 8-bit grayscale, using alpha as stroke mask."""
    if rgba.ndim == 2:
        return rgba.astype(np.uint8)
    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        # Convert RGB to grayscale then multiply by alpha
        rgb_gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        result   = (rgb_gray * alpha).astype(np.uint8)
        return result
    return cv2.cvtColor(rgba, cv2.COLOR_RGB2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Bilateral filter: smooths noise but keeps stroke edges sharp.
    Much better than Gaussian for handwriting (preserves corners/tips).
    """
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold → clean black/white binary image."""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _close_gaps(binary: np.ndarray) -> np.ndarray:
    """
    Morphological closing with a small kernel.
    Fills tiny holes inside strokes caused by fast drawing.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _crop_digit(binary: np.ndarray, padding: int = 4) -> np.ndarray | None:
    """Find bounding box of white pixels and crop with padding."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    H, W = binary.shape

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, W)
    y2 = min(y + h + padding, H)
    return binary[y1:y2, x1:x2]


def _resize_to_20(cropped: np.ndarray) -> np.ndarray:
    """
    Scale longest side to 20 px, preserving aspect ratio.
    INTER_AREA for shrink (avoids moiré), INTER_CUBIC for grow (smooth).

    Anti-aliased grayscale edges are KEPT (not re-binarized) — this is
    exactly what makes authentic MNIST digits look like soft-edged
    pixel blocks instead of harsh jagged stair-steps.
    """
    h, w = cropped.shape
    if h == 0 or w == 0:
        return np.zeros((20, 20), dtype=np.uint8)

    if h >= w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_w, new_h = 20, max(1, round(h * 20 / w))

    interp  = cv2.INTER_AREA if max(h, w) > 20 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)
    return resized


def _center_on_28x28(small: np.ndarray) -> np.ndarray:
    """
    Place digit on 28×28 frame, then shift so centre-of-mass = (14, 14).
    Matches the original MNIST preprocessing exactly (LeCun et al. 1998).

    Grayscale anti-aliasing from the sub-pixel shift is KEPT — this soft
    edge shading is exactly what real MNIST digits have, and is what
    makes the zoomed-in pixel grid look like proper MNIST blocks instead
    of a harshly-jagged silhouette.
    """
    sh, sw = small.shape

    # Geometric centre paste
    r_off = (28 - sh) // 2
    c_off = (28 - sw) // 2

    tmp = np.zeros((28, 28), dtype=np.float32)
    tmp[r_off:r_off + sh, c_off:c_off + sw] = small.astype(np.float32)

    # Compute centre-of-mass and shift to (14, 14)
    cy, cx    = ndimage.center_of_mass(tmp)
    shift_r   = round(14.0 - cy)
    shift_c   = round(14.0 - cx)
    canvas    = ndimage.shift(tmp, shift=(shift_r, shift_c),
                               order=1, mode="constant", cval=0)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def _contrast_stretch(img: np.ndarray) -> np.ndarray:
    """
    Stretch pixel values so the brightest pixel = 255.
    Ensures the digit is always full-white, not grey, for clear visibility.
    """
    mx = img.max()
    if mx == 0:
        return img
    if mx < 255:
        img = (img.astype(np.float32) * (255.0 / mx)).astype(np.uint8)
    return img


# ────────────────────────── public API ──────────────────────────────────────

class MNISTProcessor:
    """Full MNIST preprocessing pipeline with per-step intermediate images."""

    def __init__(self) -> None:
        self.step_grayscale: np.ndarray | None = None
        self.step_denoised:  np.ndarray | None = None
        self.step_binary:    np.ndarray | None = None
        self.step_closed:    np.ndarray | None = None
        self.step_cropped:   np.ndarray | None = None
        self.step_resized:   np.ndarray | None = None
        self.result:         np.ndarray | None = None

    def process(self, canvas_image: np.ndarray) -> np.ndarray | None:
        """
        Run the full pipeline on a raw RGBA canvas image.

        Returns
        -------
        np.ndarray (28, 28) uint8  or  None if no digit detected.
        """
        # 1 – grayscale
        gray = _to_grayscale(canvas_image)
        self.step_grayscale = gray.copy()

        # 2 – edge-preserving denoise
        denoised = _denoise(gray)
        self.step_denoised = denoised.copy()

        # 3 – binarize
        binary = _binarize(denoised)
        self.step_binary = binary.copy()

        # 4 – close small gaps
        closed = _close_gaps(binary)
        self.step_closed = closed.copy()

        # 5 – crop
        cropped = _crop_digit(closed)
        if cropped is None:
            self.step_cropped = self.step_resized = self.result = None
            return None
        self.step_cropped = cropped.copy()

        # 6 – resize to 20 px (re-binarized inside)
        resized = _resize_to_20(cropped)
        self.step_resized = resized.copy()

        # 7 – place on 28×28 + CoM centering + final binarize
        mnist_img = _center_on_28x28(resized)

        # 8 – contrast stretch so digit is always pure white
        mnist_img = _contrast_stretch(mnist_img)

        self.result = mnist_img
        return mnist_img

    # ── convenience ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if self.result is None:
            return {}
        arr = self.result
        return {
            "shape":          arr.shape,
            "dtype":          str(arr.dtype),
            "min":            int(arr.min()),
            "max":            int(arr.max()),
            "mean":           round(float(arr.mean()), 4),
            "nonzero_pixels": int(np.count_nonzero(arr)),
        }

    def to_pil(self) -> Image.Image | None:
        if self.result is None:
            return None
        return Image.fromarray(self.result, mode="L")

    def to_numpy_bytes(self) -> bytes | None:
        if self.result is None:
            return None
        import io
        buf = io.BytesIO()
        np.save(buf, self.result)
        return buf.getvalue()

    def to_png_bytes(self) -> bytes | None:
        pil = self.to_pil()
        if pil is None:
            return None
        import io
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
