"""
classifier.py
─────────────
Upgraded MNIST digit classifier using a Feed-Forward Neural Network 
(Multilayer Perceptron) built with TensorFlow/Keras. Trains efficiently 
on first launch with normalized data, then caches locally to ensure 
lightning-fast execution and zero memory overhead.
"""

from __future__ import annotations

import os
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# Defensive: if this module ever gets imported before app.py sets these
# (e.g. run standalone, or import order changes), set them here too so
# TensorFlow doesn't spin up its own multi-threaded pool inside a process
# that Streamlit is already running multi-threaded.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Save the trained deep learning weights as a standard Keras model file
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_model.keras")

# `DigitClassifier` is wrapped in @st.cache_resource in app.py, which means
# ONE shared instance (and one shared TF model) serves every concurrent
# Streamlit session — and each session runs in its own thread inside the
# same process. Calling model.predict()/model() concurrently on the same
# Keras model from multiple threads with no synchronization is a
# well-documented cause of native (segfault) crashes in TensorFlow — the
# C++ runtime underneath doesn't guarantee that's safe. This lock forces
# predictions (and the one-time load/train) to happen one at a time.
_MODEL_LOCK = threading.Lock()


def _train_and_save() -> None:
    import tensorflow as tf

    logger.info("Loading authentic 28×28 MNIST data from Keras datasets...")
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    
    logger.info("Applying feature scaling (normalization) to pixel data...")
    # Scale pixel values from [0, 255] down to [0.0, 1.0] as verified in the notebook
    x_train_normalized = x_train / 255.0
    x_test_normalized = x_test / 255.0

    logger.info("Building the Feed-Forward Neural Network architecture...")
    # Matches the model from the notebook exactly: Input Flatten -> 2x Dense 128 -> Dropout -> Softmax Output
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    
    logger.info("Training the network model for 15 epochs...")
    # verbose=0 keeps terminal clutter minimal during the server process
    model.fit(
        x_train_normalized, 
        y_train, 
        epochs=15, 
        validation_data=(x_test_normalized, y_test), 
        verbose=0
    )
    
    # Save the architecture + trained weights to a compact file
    model.save(_MODEL_PATH)
    logger.info("Classifier saved successfully → %s", _MODEL_PATH)


class DigitClassifier:
    """Thin wrapper around the trained TensorFlow/Keras model pipeline."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with _MODEL_LOCK:
            # Re-check inside the lock — another thread may have already
            # loaded it while we were waiting.
            if self._model is not None:
                return
            import tensorflow as tf
            if not os.path.exists(_MODEL_PATH):
                _train_and_save()
            self._model = tf.keras.models.load_model(_MODEL_PATH)

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

        # 1. Normalize the image array to the 0-1 range exactly like the training setup
        x = mnist_image.astype(np.float32) / 255.0

        # 2. Reshape to match the expected batch input shape: (1, 28, 28)
        x = np.expand_dims(x, axis=0)

        # 3. Run inference. Serialized via the lock since this model is
        # shared (via st.cache_resource) across every concurrent Streamlit
        # session/thread — concurrent unsynchronized calls into the same
        # TF model are a known crash risk. `model(x, training=False)` is
        # also used instead of `model.predict(x)`: for single-sample
        # inference it skips the extra tf.data/callback machinery that
        # `.predict()` sets up on every call, which is both faster and has
        # less surface area for issues under repeated rapid calls.
        with _MODEL_LOCK:
            predictions = self._model(x, training=False).numpy()

        digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][digit])

        return digit, confidence