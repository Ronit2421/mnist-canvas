"""
classifier.py
─────────────
Fast MNIST digit classifier used for validation only — checks whether
the digit a user drew matches the one the session is asking for.

Uses sklearn's built-in load_digits (8×8 thumbnails, upscaled to 28×28)
with a RandomForestClassifier. No internet download required, trains in
~3-5 seconds on first launch, then cached to mnist_clf.joblib.
"""

from __future__ import annotations

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_clf.joblib")


def _train_and_save() -> None:
    import joblib
    import cv2
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    logger.info("Training digit classifier (one-time, ~5 seconds) …")
    digits = load_digits()
    X_28 = np.array([
        cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR).flatten()
        for img in digits.images
    ], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_28, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    logger.info("Classifier validation accuracy: %.4f", acc)
    joblib.dump(pipe, _MODEL_PATH)
    logger.info("Classifier saved → %s", _MODEL_PATH)


class DigitClassifier:
    """Thin wrapper around the trained sklearn pipeline."""

    def __init__(self) -> None:
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        import joblib
        if not os.path.exists(_MODEL_PATH):
            _train_and_save()
        self._pipe = joblib.load(_MODEL_PATH)

    def is_ready(self) -> bool:
        return os.path.exists(_MODEL_PATH)

    def predict(self, mnist_image: np.ndarray) -> tuple[int, float]:
        """
        Predict the digit in a 28×28 uint8 image.

        Returns
        -------
        predicted_digit : int
        confidence      : float  (0–1)
        """
        self._ensure_loaded()
        x = mnist_image.astype(np.float32).flatten().reshape(1, -1)
        probs = self._pipe.predict_proba(x)[0]
        digit = int(np.argmax(probs))
        return digit, float(probs[digit])
