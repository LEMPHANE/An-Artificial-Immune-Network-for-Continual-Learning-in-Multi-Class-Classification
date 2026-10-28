"""
Main window: wires the generated layout [Ui_MainWindow in widgets.py] to the model, the data
store, the training thread and the four viewer pages [also in widgets.py].
"""
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QHBoxLayout, QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QStackedWidget, QVBoxLayout,
)

from settings import EPOCHS, UNKNOWN_DISTANCE, make_model
from data.data_store import DataStore, class_name
from src.workers import TrainWorker
from ui.draw_canvas import DrawScene
from ui.plots import HistoryCanvas, short_name
from ui.widgets import LoaderDialog, MemoryView, MetricsView, TestView, TrainingView, Ui_MainWindow

# Which style the drawn image is converted to: "digit" | "apparel" | "raw" (see draw_canvas.py)
DRAW_MODE = "digit"

VIEW_TEST, VIEW_TRAINING, VIEW_MEMORY, VIEW_METRICS = range(4)
VIEW_NAMES = ("Test", "Training", "Memory", "Metrics")
# Short on purpose: the info line only has the room left of the OK / Restart / Cancel / Back buttons.
VIEW_HINTS = {
    VIEW_TRAINING: "Live view of the current training run",
    VIEW_MEMORY: "Hover: inspect · click: pin · legend: hide / isolate",
    VIEW_METRICS: "Rows: true class · columns: predicted class",
}
TEST_HINT = "Train, press Draw, sketch a digit, then OK"
DRAW_HINT = "Draw with the left mouse button · OK classifies"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # ---- state --------------------------------------------------------
        self.model = make_model()
        self.worker = None                  # TrainWorker while a run is in progress
        self.drawn_image = None             # (28, 28) float32 in [0, 1]; set when the user confirms a drawing
        self._mem = None                    # (mem_vecs, mem_cls) snapshot sent by the worker (GUI-thread copy)
        self._memory_dirty = False
        self._prev_view = VIEW_TEST
        self._test_hint = TEST_HINT
        self._cls_info = {}                 # last class_started payload (for the status line)

        # ---- data -------------------------------------------------------
        self.store = DataStore(max_train_per_class=None)
        self.store.progress.connect(self.ui.statusbar.showMessage)
        self.store.ready.connect(lambda: self.ui.statusbar.showMessage("Datasets ready", 4000))
        self.store.failed.connect(lambda m: self.ui.statusbar.showMessage(f"Dataset loading failed: {m}"))
        self.store.start_loading()

        # ---- layout + pages ---------------------------------------------
        self._build_layout()
        self._build_views()

        # ---- drawing ----------------------------------------------------
        self.draw_scene = DrawScene(self)
        view = self.ui.graphicsView
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setScene(self.draw_scene)      # blank canvas visible from the start (not drawable until Draw)
        self._viewport = view.viewport()    # kept as a Python reference: the filter must not touch the view at exit
        self._viewport.installEventFilter(self)  # keep the canvas fitted to the viewer

        # ---- buttons ----------------------------------------------------
        ui = self.ui
        ui.pushButton.clicked.connect(self.open_loader)          # Load
        ui.pushButton_2.clicked.connect(self.start_training)     # Train
        ui.pushButton_5.setCheckable(True)                       # Draw (toggle)
        ui.pushButton_5.toggled.connect(self.toggle_draw)
        ui.pushButton_3.clicked.connect(self.confirm_drawing)    # OK: confirm + classify
        ui.pushButton_8.clicked.connect(self.restart_drawing)    # Restart: clear the canvas
        ui.pushButton_7.clicked.connect(self.go_back)            # Back: leave draw mode, previous view
        ui.pushButton_4.clicked.connect(self.reset_all)          # RESET
        ui.pushButton_6.setEnabled(False)                        # Cancel: not wired yet
        ui.pushButton_6.setToolTip("Not wired yet")

        self._reset_labels()
        self._update_summary()
        self._update_buttons()
        self.show_view(VIEW_TEST)

    # ===================================================================
    # Layout built in code (no .ui edits)
    # ===================================================================
    def _build_layout(self):
        ui = self.ui

        root = QVBoxLayout(ui.centralwidget)
        root.setContentsMargins(24, 16, 24, 12)
        root.addWidget(ui.verticalLayoutWidget_6)
        self.resize(1120, 840)
        self.setMinimumSize(980, 720)
        ui.body.setStretch(0, 5)   # viewer row (includes the title)
        ui.body.setStretch(1, 3)   # history row

        # small info line under the viewer: takes the free space (clipped if long) instead of widening the window
        row = ui.horizontalLayout_2
        row.takeAt(1)              # the expanding spacer after label_10
        row.setStretch(0, 1)
        ui.label_10.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        # right panel: pack the labels at the top; the class list may wrap
        ui.right_panel.addStretch(1)
        ui.label_11.setWordWrap(True)

    def _build_views(self):
        ui = self.ui

        # ---- stacked viewer: Test (the existing graphicsView) | Training | Memory | Metrics
        self.test_view = TestView(ui.graphicsView)
        self.training = TrainingView()
        self.memory = MemoryView()
        self.metrics = MetricsView()

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        for page in (self.test_view, self.training, self.memory, self.metrics):
            self.stack.addWidget(page)

        nav = QHBoxLayout()
        nav.setSpacing(2)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for i, name in enumerate(VIEW_NAMES):
            b = QPushButton(name)
            b.setObjectName("navButton")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(b, i)
            nav.addWidget(b)
        nav.addStretch(1)
        self.nav.idClicked.connect(self.show_view)

        mp = ui.middle_panel                  # [label_8, graphicsView, horizontalLayout_2]
        mp.removeWidget(ui.graphicsView)      # it now lives inside the Test page
        mp.insertLayout(1, nav)
        mp.insertWidget(2, self.stack, 1)

        # ---- footer: per-class history plot replaces graphicsView_2
        self.history_canvas = HistoryCanvas()
        fl = ui.horizontalLayout              # [spacer, graphicsView_2, spacer, details]
        i = fl.indexOf(ui.graphicsView_2)
        fl.removeWidget(ui.graphicsView_2)
        ui.graphicsView_2.hide()
        fl.insertWidget(i, self.history_canvas, 6)
        self.history_canvas.setMinimumHeight(150)
        fl.setStretch(0, 0)                   # the two padding spacers keep their minimum width,
        fl.setStretch(2, 0)                   # so the plot gets the room

    # ===================================================================
    # Views
    # ===================================================================
    def show_view(self, index):
        current = self.stack.currentIndex()
        if index != current:
            self._prev_view = current
            if current == VIEW_TEST and self.ui.pushButton_5.isChecked():
                self.ui.pushButton_5.setChecked(False)       # leaving Test ends draw mode
            self.stack.setCurrentIndex(index)
        self.nav.button(index).setChecked(True)
        self.ui.label_10.setText(self._test_hint if index == VIEW_TEST else VIEW_HINTS[index])
        if index == VIEW_MEMORY and self._memory_dirty:
            self._refresh_memory()

    def go_back(self):
        if self.ui.pushButton_5.isChecked():
            self.ui.pushButton_5.setChecked(False)
        if self._prev_view != self.stack.currentIndex():
            self.show_view(self._prev_view)

    def _set_test_hint(self, text):
        self._test_hint = text
        if self.stack.currentIndex() == VIEW_TEST:
            self.ui.label_10.setText(text)

    # The memory views are fed with the worker's copies; the PCA is fitted on every loaded antigen
    # so its axes move when classes are loaded, not when a class finishes training.
    def _touch_memory(self):
        self._memory_dirty = True
        if self.stack.currentIndex() == VIEW_MEMORY:
            self._refresh_memory()

    def _refresh_memory(self):
        self._memory_dirty = False
        if self._mem is None:
            self.memory.clear()
            return
        vecs, cls = self._mem
        x, y = self.store.train_data()
        self.memory.set_memory(vecs, cls, x.reshape(len(x), -1), y, self.model.memory_size)

    # ===================================================================
    # Loading
    # ===================================================================
    def open_loader(self):
        if self.worker is not None:
            return
        dialog = LoaderDialog(self.store, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        labels = dialog.selected_labels()
        if labels:
            self._load_labels(labels)

    def _load_labels(self, labels):
        for label in labels:
            self.store.load_class(label)
        names = ", ".join(class_name(l) for l in labels)
        total = sum(len(self.store.loaded[l].x_train) for l in self.store.loaded)
        self.ui.statusbar.showMessage(
            f"Loaded: {names}  |  classes in memory: {len(self.store.loaded)}, train samples: {total}", 8000
        )
        self._touch_memory()
        self._update_summary()
        self._update_buttons()

    # ===================================================================
    # Training
    # ===================================================================
    def start_training(self):
        if self.worker is not None:
            return
        if not self.store.pending_labels():
            self.ui.statusbar.showMessage("Nothing to train - load a class first", 5000)
            return
        if self.ui.pushButton_5.isChecked():
            self.ui.pushButton_5.setChecked(False)

        w = TrainWorker.from_store(self.model, self.store, EPOCHS, parent=self)
        w.train_started.connect(self._on_train_started)
        w.class_started.connect(self._on_class_started)
        w.progress.connect(self.ui.progressBar.setValue)
        w.epoch_done.connect(self.training.on_epoch_done)
        w.memory_updated.connect(self._on_memory_updated)
        w.class_done.connect(self._on_class_done)
        w.evaluated.connect(self._on_evaluated)
        w.history_updated.connect(self.history_canvas.update_history)
        w.train_ended.connect(self._on_train_ended)
        w.stopped.connect(lambda: self.training.note("Training stopped"))
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_worker_finished)   # always fires, after every other signal

        self.worker = w
        self._update_buttons()
        self.show_view(VIEW_TRAINING)
        w.start()

    def _on_train_started(self, classes, total_steps):
        bar = self.ui.progressBar
        bar.setRange(0, max(total_steps, 1))
        bar.setValue(0)
        bar.setFormat("%p%")
        self.training.on_train_started(classes, total_steps)
        self.ui.statusbar.showMessage(f"Training {len(classes)} class(es): " + ", ".join(short_name(c) for c in classes))

    def _on_class_started(self, info):
        self._cls_info = info
        self.ui.progressBar.setFormat(f"{short_name(info['cls'])}  %p%")
        self.training.on_class_started(info)

    def _on_memory_updated(self, vecs, cls):
        self._mem = (vecs, cls)

    def _on_class_done(self, cls, memory_size):
        self.store.mark_trained(cls)          # only now is the class really in the model
        self.training.on_class_done(cls, memory_size)
        self._update_summary()
        self._touch_memory()

    def _on_evaluated(self, m):
        self.training.on_evaluated(m)
        self.metrics.update_metrics(m)
        ui = self.ui
        ui.label_12.setText(f"Total Accuracy: {m['accuracy']:.1%}")
        ui.label_13.setText(f"Balanced accuracy: {m['balanced_accuracy']:.1%}")
        ui.label_15.setText(f"New class accuracy: {m['new_class_recall']:.1%}")
        ui.label_15.setToolTip(f"Recall on the class just added ({class_name(m['new_class'])})")

    def _on_train_ended(self, trained):
        self.ui.progressBar.setFormat("Done")
        size = 0 if self._mem is None else len(self._mem[0])
        msg = f"Training finished: {len(trained)} class(es) in the model, {size} memory cells"
        if len(trained) < 2:
            msg += " (load another class to see evaluation)"
        self.training.note(msg)
        self.ui.statusbar.showMessage(msg, 10000)

    def _on_failed(self, msg):
        self.ui.progressBar.setFormat("Failed")
        self.training.note(f"Training failed: {msg}")
        QMessageBox.warning(self, "Training failed", msg)

    def _on_worker_finished(self):
        w, self.worker = self.worker, None
        if w is not None:
            w.deleteLater()
        self._update_summary()
        self._update_buttons()

    # ===================================================================
    # Drawing + classification
    # ===================================================================
    def toggle_draw(self, on):
        view = self.ui.graphicsView
        if on:
            self.show_view(VIEW_TEST)
            self._clear_prediction()
            self.draw_scene.begin_drawing()
            view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self._fit_canvas()
            self._set_test_hint(DRAW_HINT)
            self.ui.statusbar.showMessage("Draw mode: draw with the left mouse button, OK to classify, Restart to clear")
        else:
            self.draw_scene.end_drawing()
            view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self._update_buttons()

    def confirm_drawing(self):
        if not self.draw_scene.drawing_enabled:
            return  # OK only means "confirm + classify" while in draw mode
        if self.model.knn is None:
            self.ui.statusbar.showMessage("Train the model first - there is nothing to classify with yet", 5000)
            return
        arr = self.draw_scene.to_array(DRAW_MODE)
        if arr is None:
            self.ui.statusbar.showMessage("The canvas is empty - draw something first", 4000)
            return
        self.drawn_image = arr
        self.draw_scene.show_array(arr)         # show exactly what the model will see
        self.ui.pushButton_5.setChecked(False)  # leaves draw mode
        self.classify(arr)

    def classify(self, arr):
        info = self.model.explain(arr, unknown_dist=UNKNOWN_DISTANCE)
        self.test_view.prediction.show_result(info)

        name = class_name(info["label"])
        p = info["proba"][info["label"]]
        if info["unknown"]:
            hint = f"Unknown · best guess {name}"
            msg = f"Unknown (best guess {name}, {info['nearest']:.2f} from the nearest memory cell)"
        else:
            hint = f"Prediction: {name} ({p:.0%}) · nearest cell {info['nearest']:.1f}"
            msg = f"Prediction: {name} ({p:.0%} of the nearest cells), nearest memory cell {info['nearest']:.2f} away"
        self._set_test_hint(hint)
        self.ui.statusbar.showMessage(msg, 8000)

        # mark the drawing on the PCA map (after it is up to date: refreshing it clears the probe)
        if self._memory_dirty:
            self._refresh_memory()
        self.memory.pca.set_probe(arr, info["neighbor_vecs"], info["neighbor_cls"])

    def restart_drawing(self):
        self.draw_scene.clear()
        self._clear_prediction()
        if self.draw_scene.drawing_enabled:
            self._set_test_hint(DRAW_HINT)

    def _clear_prediction(self):
        self.drawn_image = None
        self.test_view.prediction.clear()
        self.memory.pca.clear_probe()

    def _fit_canvas(self):
        try:
            view = self.ui.graphicsView
            if view.scene() is self.draw_scene:
                view.fitInView(self.draw_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        except RuntimeError:
            pass    # the view is already gone (window closing)

    def eventFilter(self, obj, event):
        if obj is self._viewport and event.type() == QEvent.Type.Resize:
            self._fit_canvas()
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_canvas()

    # ===================================================================
    # Reset
    # ===================================================================
    def _confirm_reset(self):
        answer = QMessageBox.question(
            self, "Reset everything?",
            "This unloads every class and wipes the trained model, its history and all plots.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def reset_all(self):
        if self.worker is not None or not self._confirm_reset():
            return
        if self.ui.pushButton_5.isChecked():
            self.ui.pushButton_5.setChecked(False)

        self.model.reset()
        self.store.reset()
        self._mem = None
        self._memory_dirty = False
        self.draw_scene.clear()
        self._clear_prediction()
        self.history_canvas.clear()
        self.training.reset()
        self.memory.clear()
        self.metrics.clear()
        self.ui.progressBar.setRange(0, 1)
        self.ui.progressBar.setValue(0)
        self._reset_labels()
        self._set_test_hint(TEST_HINT)
        self._update_summary()
        self._update_buttons()
        self.ui.statusbar.showMessage("Everything was reset", 5000)

    # ===================================================================
    # Labels + button states
    # ===================================================================
    def _reset_labels(self):
        ui = self.ui
        ui.progressBar.setRange(0, 1)
        ui.progressBar.setValue(0)
        ui.progressBar.setFormat("Idle")
        ui.label_12.setText("Total Accuracy: -")
        ui.label_13.setText("Balanced accuracy: -")
        ui.label_15.setText("New class accuracy: -")
        ui.label_15.setToolTip("")

    def _update_summary(self):
        ui = self.ui
        trained = list(self.model.trained_classes)
        pending = len(self.store.pending_labels())
        size = 0 if self._mem is None else len(self._mem[0])
        ui.label_5.setText(f"Memory size: {size}")
        ui.label_6.setText(f"Num Classes: {len(trained)}" + (f" (+{pending} waiting)" if pending else ""))
        ui.label_11.setText("Classes Introduced: " + (", ".join(short_name(c) for c in trained) or "none yet"))

    def _update_buttons(self):
        ui = self.ui
        training = self.worker is not None
        drawing = ui.pushButton_5.isChecked()
        has_anything = bool(self.store.loaded) or self.model.knn is not None
        ui.pushButton.setEnabled(not training)                                      # Load
        ui.pushButton_2.setEnabled(not training and bool(self.store.pending_labels()))  # Train
        ui.pushButton_5.setEnabled(not training)                                    # Draw
        ui.pushButton_3.setEnabled(drawing and not training)                        # OK
        ui.pushButton_4.setEnabled(not training and has_anything)                   # RESET

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait(5000)
        super().closeEvent(event)
