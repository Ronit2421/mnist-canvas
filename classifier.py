"""
classifier.py
─────────────
Upgraded MNIST digit classifier using real 28×28 shapes from OpenML 
via a K-Neighbors Classifier. Trains efficiently on first launch,
then caches locally to ensure lightning-fast execution.
"""

from __future__ import annotations

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_clf.joblib")


def _train_and_save() -> None:
    import joblib
    from sklearn.datasets import fetch_openml
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    logger.info("Fetching authentic 28×28 MNIST data from OpenML (one-time download)...")
    
    # Fetch a subset of the real MNIST dataset to keep compilation fast
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    y = y.astype(np.int32)
    
    # Use 15,000 samples for a perfect balance between high accuracy and fast caching
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=15000, stratify=y, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.15, random_state=42, stratify=y_sub
    )

    logger.info("Training a K-Neighbors Classifier for structural stroke validation...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)),
    ])
    
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    logger.info("Real-MNIST KNN Classifier validation accuracy: %.4f", acc)
    
    joblib.dump(pipe, _MODEL_PATH)
    logger.info("Classifier saved successfully → %s", _MODEL_PATH)


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
        Returns:
            predicted_digit : int
            confidence      : float (0–1)
        """
        self._ensure_loaded()
        x = mnist_image.astype(np.float32).flatten().reshape(1, -1)
        
        # Predict digit label
        digit = int(self._pipe.predict(x)[0])
        
        # Calculate neighbor distance ratios to output a proxy confidence score
        probs = self._pipe.predict_proba(x)[0]
        confidence = float(probs[digit])
        
        return digit, confidence