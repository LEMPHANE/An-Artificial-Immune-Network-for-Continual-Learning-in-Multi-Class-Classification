"""
Dataset handling for the GUI: MNIST (labels 0-9) + Fashion-MNIST (labels 10-19).

> Fetched with sklearn's fetch_openml in a background thread.
> Cached as a single compressed .npz in /data/cache so later startups are fast.
> Individual classes are pulled into "memory" on demand with DataStore.load_class().
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
FASHION_OFFSET = 10  # Fashion-MNIST label k  ->  k + 10

MNIST_NAMES = [str(d) for d in range(10)]
FASHION_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CLASS_NAMES = {i: f"MNIST: {n}" for i, n in enumerate(MNIST_NAMES)}
CLASS_NAMES.update({FASHION_OFFSET + i: f"Fashion: {n}" for i, n in enumerate(FASHION_NAMES)})


def class_name(label):
    """Numeric model label -> readable name ('MNIST: 5', 'Fashion: Sneaker')."""
    return CLASS_NAMES.get(int(label), str(label))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "datasets.npz"


@dataclass
class ClassData:
    label: int
    name: str
    x_train: np.ndarray  # (n, 28, 28) float32 in [0, 1]
    y_train: np.ndarray  # (n,) int
    x_test: np.ndarray
    y_test: np.ndarray


# -------------------------------------------------------------------
# Background loader
# -------------------------------------------------------------------
class _LoadThread(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)  # dict of arrays
    failed = pyqtSignal(str)

    def run(self):
        try:
            arrays = None
            if CACHE_FILE.is_file():
                try:
                    self.progress.emit("Loading cached datasets...")
                    with np.load(CACHE_FILE) as f:
                        arrays = {k: f[k] for k in f.files}
                except Exception:
                    arrays = None  # corrupt cache -> re-download

            if arrays is None:
                arrays = self._download()
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = CACHE_FILE.with_name("datasets.tmp.npz")
                np.savez_compressed(tmp, **arrays)
                tmp.replace(CACHE_FILE)

            self.done.emit(arrays)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _download(self):
        from sklearn.datasets import fetch_openml

        home = str(CACHE_DIR / "openml")
        parts = {}
        for key, name in (("mnist", "mnist_784"), ("fashion", "Fashion-MNIST")):
            self.progress.emit(f"Downloading {name} from OpenML (first run only)...")
            x, y = fetch_openml(name, version=1, as_frame=False, return_X_y=True, data_home=home)
            x = x.astype(np.uint8).reshape(-1, 28, 28)
            y = y.astype(np.int64)
            # standard split: first 60k train, last 10k test
            parts[key] = (x[:60000], y[:60000], x[60000:], y[60000:])

        mx_tr, my_tr, mx_te, my_te = parts["mnist"]
        fx_tr, fy_tr, fx_te, fy_te = parts["fashion"]
        return {
            "x_train": np.concatenate([mx_tr, fx_tr]),
            "y_train": np.concatenate([my_tr, fy_tr + FASHION_OFFSET]),
            "x_test": np.concatenate([mx_te, fx_te]),
            "y_test": np.concatenate([my_te, fy_te + FASHION_OFFSET]),
        }


# -------------------------------------------------------------------
# Store shared by the main window and the loader dialog
# -------------------------------------------------------------------
class DataStore(QObject):
    progress = pyqtSignal(str)
    ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, max_train_per_class=None, max_test_per_class=None, seed=0, parent=None):
        super().__init__(parent)
        self.max_train_per_class = max_train_per_class  # None = all (~6000/class)
        self.max_test_per_class = max_test_per_class    # None = all (~1000/class)
        self.seed = seed

        self._data = None          # dict of the four big uint8/int arrays
        self._index = {}           # split -> label -> indices
        self._icons = {}           # label -> QIcon
        self.loaded = {}           # label -> ClassData  (the "in memory" classes, in load order)
        self.trained = set()       # labels the model has finished training on
        self._thread = None
        self.error = None

    # ---- loading ----------------------------------------------------
    def start_loading(self):
        if self._thread is not None:
            return
        self._thread = _LoadThread(self)
        self._thread.progress.connect(self.progress)
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_done(self, arrays):
        self._data = arrays
        for split in ("train", "test"):
            y = arrays[f"y_{split}"]
            self._index[split] = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}
        for label in CLASS_NAMES:
            self._icons[label] = self._make_icon(label)
        self.ready.emit()

    def _on_failed(self, msg):
        self.error = msg
        self.failed.emit(msg)

    def is_ready(self):
        return self._data is not None

    # ---- icons ------------------------------------------------------
    def _make_icon(self, label):
        idx = self._index["train"][label][0]
        img = np.ascontiguousarray(self._data["x_train"][idx], dtype=np.uint8)
        qimg = QImage(img.data, 28, 28, 28, QImage.Format.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        return QIcon(pix)

    def icon(self, label):
        return self._icons.get(label, QIcon())

    # ---- per-class "memory" ----------------------------------------
    def is_loaded(self, label):
        return label in self.loaded

    def loaded_labels(self):
        return sorted(self.loaded)

    # ---- training state ---------------------------------------------
    def is_trained(self, label):
        return label in self.trained

    def mark_trained(self, label):
        self.trained.add(int(label))

    def pending_labels(self):
        """Loaded but not yet trained classes, in the order they were loaded."""
        return [label for label in self.loaded if label not in self.trained]

    def _pick(self, split, label, cap):
        idx = self._index[split][label]
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(self.seed + label)
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        return idx

    def load_class(self, label):
        """Pull one class into memory (capped, normalised to [0, 1])."""
        if not self.is_ready():
            raise RuntimeError("Datasets are not loaded yet")
        if label in self.loaded:
            return self.loaded[label]

        tr = self._pick("train", label, self.max_train_per_class)
        te = self._pick("test", label, self.max_test_per_class)
        d = self._data
        cd = ClassData(
            label=label,
            name=CLASS_NAMES[label],
            x_train=d["x_train"][tr].astype(np.float32) / 255.0,
            y_train=d["y_train"][tr].copy(),
            x_test=d["x_test"][te].astype(np.float32) / 255.0,
            y_test=d["y_test"][te].copy(),
        )
        self.loaded[label] = cd
        return cd

    def pending_data(self):
        """(x, y, order) for the loaded-but-untrained classes, in load order.

        Fresh concatenated copies, so it is safe to hand them to a worker thread.
        `order` is the class order to pass to CLONALG.partial_fit(class_order=...).
        """
        order = self.pending_labels()
        if not order:
            return np.empty((0, 28, 28), np.float32), np.empty((0,), np.int64), []
        items = [self.loaded[label] for label in order]
        return (np.concatenate([c.x_train for c in items]),
                np.concatenate([c.y_train for c in items]),
                order)

    def train_data(self):
        """(x, y) for every loaded class - ready for CLONALG.fit()."""
        return self._concat("x_train", "y_train")

    def eval_data(self):
        """(x_test, y_test) for every loaded class - ready for fit(eval_data=...)."""
        return self._concat("x_test", "y_test")

    def _concat(self, xk, yk):
        if not self.loaded:
            return np.empty((0, 28, 28), np.float32), np.empty((0,), np.int64)
        items = [self.loaded[k] for k in sorted(self.loaded)]
        return (np.concatenate([getattr(c, xk) for c in items]),
                np.concatenate([getattr(c, yk) for c in items]))

    # ---- reset -------------------------------------------------------
    def reset(self):
        """Unload every class and forget what was trained.

        The raw datasets, index and icons stay in memory - re-downloading or
        rebuilding them would make Reset needlessly slow.
        """
        self.loaded.clear()
        self.trained.clear()
