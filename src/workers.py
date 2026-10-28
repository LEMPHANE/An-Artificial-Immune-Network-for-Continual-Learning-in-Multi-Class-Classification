"""
Background training for the GUI.

TrainWorker runs CLONALG.partial_fit() on a QThread so the window stays responsive,
and turns the model's callback events into Qt signals. Signals emitted from the
worker thread are delivered to slots on the GUI thread (queued connections), so
slots can touch widgets directly.

While a worker is running the model is being mutated: the main window must keep
Load / Train / Draw-and-predict / Reset disabled and not read model.mem_vecs
until `finished` fires.

Typical use (main window):

    worker = TrainWorker.from_store(self.model, self.store, EPOCHS, parent=self)
    worker.class_done.connect(lambda cls, size: self.store.mark_trained(cls))
    worker.progress.connect(...)
    worker.finished.connect(self.on_training_finished)   # always fires, last
    worker.start()
"""
import traceback

from PyQt6.QtCore import QThread, pyqtSignal


class TrainWorker(QThread):
    # --- signals (payload keys match the model's callback events) -----------------
    train_started = pyqtSignal(list, int)  # classes in training order, total_steps
    class_started = pyqtSignal(dict)       # cls, k, n_classes, population, n_antigens, batch_size
    progress = pyqtSignal(int, int)        # step, total_steps   (one per batch)
    epoch_done = pyqtSignal(dict)          # cls, epoch, epochs, nearest, pop_start, pop_end,
                                           #   clones_added, pruned
    memory_updated = pyqtSignal(object, object)  # (mem_vecs, mem_cls) copies, emitted just before class_done
    class_done = pyqtSignal(int, int)      # cls, memory pool size  (class is now in memory)
    evaluated = pyqtSignal(dict)           # metrics dict from CLONALG.evaluate_stage
    history_updated = pyqtSignal(dict)     # snapshot of model.history after each eval (thread-safe copy)
    train_ended = pyqtSignal(list)         # every class trained so far (all pending finished)
    stopped = pyqtSignal()                 # cancelled via request_stop()
    failed = pyqtSignal(str)               # "ValueError: ..." (traceback goes to the console)
    # QThread.finished (built in) fires after any of the above, always last.

    def __init__(self, model, antigens, labels, eval_data, class_order, epochs, parent=None):
        super().__init__(parent)
        self.model = model
        self._x = antigens
        self._y = labels
        self._eval = eval_data
        self._order = list(class_order)
        self.epochs = epochs
        self._stop_requested = False

    @classmethod
    def from_store(cls, model, store, epochs, parent=None):
        """Snapshot the store's pending classes (load order) + test data of everything loaded."""
        x, y, order = store.pending_data()
        if not order:
            raise ValueError("No loaded classes are waiting to be trained")
        return cls(model, x, y, store.eval_data(), order, epochs, parent)

    # ---- control ------------------------------------------------------------------
    def request_stop(self):
        """Stop at the next batch; the class in progress is discarded, finished ones are kept."""
        self._stop_requested = True
        self.model.request_stop()

    # ---- thread body --------------------------------------------------------------
    def run(self):
        previous = self.model.callback
        self.model.callback = self._on_event
        try:
            self.model.partial_fit(
                self._x, self._y,
                epochs=self.epochs,
                eval_data=self._eval,
                class_order=self._order,
            )
        except Exception as e:  # report to the GUI instead of dying silently in a thread
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.model.callback = previous

    def _on_event(self, event, **info):
        if event == "train_start":
            # partial_fit clears the model's stop flag right before this event, so a
            # request_stop() that arrived before the thread got going is re-applied here.
            if self._stop_requested:
                self.model.request_stop()
            self.train_started.emit(list(info["classes"]), info["total_steps"])
        elif event == "class_start":
            self.class_started.emit(dict(info))
        elif event == "batch_end":
            self.progress.emit(info["step"], info["total_steps"])
        elif event == "epoch_end":
            self.epoch_done.emit(dict(info))
        elif event == "class_end":
            # copies taken in this thread: the GUI never reads arrays the model may replace next
            self.memory_updated.emit(self.model.mem_vecs.copy(), self.model.mem_cls.copy())
            self.class_done.emit(info["cls"], info["memory_size"])
        elif event == "eval":
            self.evaluated.emit(info["metrics"])
            # copied here, in the thread that mutates it, so the GUI never reads a dict mid-update
            self.history_updated.emit({c: list(p) for c, p in self.model.history.items()})
        elif event == "train_end":
            self.train_ended.emit(list(info["trained_classes"]))
        elif event == "stopped":
            self.stopped.emit()
