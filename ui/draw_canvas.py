"""
Drawing surface for the viewer.

DrawScene is a QGraphicsScene: attach it to the existing graphicsView with setScene().
The user draws white strokes on a 280x280 black canvas; to_array() downsamples it to a
28x28 grayscale image (float32 in [0, 1]) in the style of the chosen dataset.
"""
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsScene

CANVAS_SIZE = 280   # on-screen drawing resolution (10x the target)
PEN_WIDTH = 20      # ~2px once shrunk to 28x28, close to MNIST stroke thickness

# Preprocessing modes for the 28x28 output:
#   "digit"   - MNIST style: crop to the drawing, fit in 20x20, centre by centre of mass
#   "apparel" - Fashion-MNIST style: crop to the drawing, fit the longest side to 28, centre by bounding box
#   "raw"     - plain 280x280 -> 28x28 downscale, no cropping or centring
MODES = ("digit", "apparel", "raw")


# -------------------------------------------------------------------
# numpy <-> QImage helpers
# -------------------------------------------------------------------
def _to_qimage(arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape
    return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _from_qimage(img):
    img = img.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)[:, :w].copy()


def _resize(arr, w, h):
    img = _to_qimage(arr).scaled(
        w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    return _from_qimage(img).astype(np.float32)


# -------------------------------------------------------------------
# Canvas -> 28x28
# -------------------------------------------------------------------
def preprocess(canvas, mode="digit"):
    """canvas: (H, W) uint8, white-on-black. Returns (28, 28) float32 in [0, 1], or None if empty."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if canvas.max() == 0:
        return None

    if mode == "raw":
        return np.clip(_resize(canvas, 28, 28) / 255.0, 0, 1).astype(np.float32)

    ys, xs = np.nonzero(canvas > 10)
    if len(ys) == 0:
        return None
    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape

    box = 20 if mode == "digit" else 28
    scale = box / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = _resize(crop, new_w, new_h)

    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized

    if mode == "digit":
        total = out.sum()
        if total > 0:
            rows, cols = np.indices(out.shape)
            cy, cx = (out * rows).sum() / total, (out * cols).sum() / total
            out = np.roll(out, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))

    return np.clip(out / 255.0, 0, 1).astype(np.float32)


# -------------------------------------------------------------------
# The scene
# -------------------------------------------------------------------
class DrawScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(0, 0, CANVAS_SIZE, CANVAS_SIZE, parent)
        self.setBackgroundBrush(QColor(0, 0, 0))

        self._image = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage.Format.Format_RGB32)
        self._image.fill(Qt.GlobalColor.black)
        self._item = self.addPixmap(QPixmap.fromImage(self._image))

        self.drawing_enabled = False
        self.preview_shown = False  # True while a confirmed 28x28 is displayed instead of strokes
        self._last = None

    # ---- public API -------------------------------------------------
    def clear(self):
        self._image.fill(Qt.GlobalColor.black)
        self._refresh()
        self.preview_shown = False
        self._last = None

    def begin_drawing(self):
        """Enter draw mode. If a confirmed preview is on screen, start from a blank canvas."""
        if self.preview_shown:
            self.clear()
        self.drawing_enabled = True

    def end_drawing(self):
        self.drawing_enabled = False
        self._last = None

    def to_array(self, mode="digit"):
        """Current drawing as a (28, 28) float32 array in [0, 1], or None if the canvas is empty."""
        return preprocess(_from_qimage(self._image), mode)

    def show_array(self, arr28):
        """Display a 28x28 array (blocky, nearest-neighbour) so the user sees what the model will see."""
        small = _to_qimage(np.clip(arr28, 0, 1) * 255)
        big = small.scaled(
            CANVAS_SIZE, CANVAS_SIZE,
            Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation,
        )
        self._image = big.convertToFormat(QImage.Format.Format_RGB32)
        self._refresh()
        self.preview_shown = True

    # ---- drawing ----------------------------------------------------
    def _refresh(self):
        self._item.setPixmap(QPixmap.fromImage(self._image))

    def _stroke(self, p1, p2):
        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.white, PEN_WIDTH, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if p1 == p2:
            painter.drawPoint(p1)
        else:
            painter.drawLine(p1, p2)
        painter.end()
        self._refresh()

    def mousePressEvent(self, ev):
        if self.drawing_enabled and ev.button() == Qt.MouseButton.LeftButton:
            self._last = QPointF(ev.scenePos())
            self._stroke(self._last, self._last)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.drawing_enabled and self._last is not None and (ev.buttons() & Qt.MouseButton.LeftButton):
            pos = QPointF(ev.scenePos())
            self._stroke(self._last, pos)
            self._last = pos
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._last is not None:
            self._last = None
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)
