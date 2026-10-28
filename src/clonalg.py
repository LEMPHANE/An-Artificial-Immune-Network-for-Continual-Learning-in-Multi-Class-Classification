# Clonal Selection Methods
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, euclidean_distances


# ===================================================================
# DEFAULT EVENT HANDLER (console)
# ===================================================================
# The model reports progress through callback(event, **info) instead of
# printing. The GUI worker passes its own callback that turns these events
# into Qt signals; scripts get this one.
#
# Events:
#   train_start  classes, total_steps
#   class_start  cls, k, n_classes, population, n_antigens, batch_size
#   batch_end    cls, epoch, epochs, batch, n_batches, step, total_steps
#   epoch_end    cls, epoch, epochs, nearest, pop_start, pop_end, clones_added, pruned
#   class_end    cls, memory_size
#   eval         metrics (dict, see evaluate_stage)
#   stopped      (training was cancelled; the class in progress is discarded)
#   train_end    trained_classes
def console_callback(event, **i):
    if event == 'train_start':
        print(f"Classes Introduced: {i['classes']}\n")
    elif event == 'class_start':
        print(
            f"==================== Class {i['cls']} "
            f"({i['k']} of {i['n_classes']}) ===================="
        )
        print(
            f"Initial population = {i['population']}, "
            f"antigens = {i['n_antigens']}, batch size = {i['batch_size']}"
        )
    elif event == 'epoch_end':
        print(
            f"Epoch {i['epoch']}/{i['epochs']}: "
            f"mean dist antigen->nearest memory cell = {i['nearest']:.3f}, "
            f"population {i['pop_start']} -> {i['pop_end']} "
            f"({i['clones_added']} clones added, {i['pruned']} pruned)"
        )
    elif event == 'class_end':
        print(f"Memory pool size after class {i['cls']}: {i['memory_size']}\n")
    elif event == 'eval':
        m = i['metrics']
        print(
            f"[EVAL] {m['stage']} classes seen: accuracy = {m['accuracy']:.4f}.. "
            f"balanced = {m['balanced_accuracy']:.4f}.. "
            f"recall on new class {m['new_class']} = {m['new_class_recall']:.4f}"
        )
        print('[EVAL] Per-class accuracy:')
        for cls, (acc, n) in m['per_class'].items():
            print(f'  class {cls}: {acc:.4f} (n={n})')
        print(f"[EVAL] Confusion matrix (rows=true, cols=pred, order={m['labels']}):")
        print(m['confusion'])
        print()
    elif event == 'stopped':
        print('Training stopped.\n')


class CLONALG:

    # ===================================================================
    # CONSTRUCTOR
    # ===================================================================
    def __init__(
        self,
        population_size=30,
        memory_size=10,
        selection_size=5,
        num_clones=3,
        mutation_rate=0.1,
        similarity_threshold=9.0,
        n_neighbors=3,
        weights='distance',
        metric='euclidean',
        batch_size=None,
        alpha_max=0.5,
        noise_std=0.05,
        callback=console_callback,
    ):
        # Model Parameters
        self.population_size = population_size
        self.memory_size = memory_size
        self.selection_size = selection_size  # number of best clones to select for cloning
        self.num_clones = num_clones            # total number of clones to generate from best clones
        self.mutation_rate = mutation_rate
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.batch_size = batch_size    # None = the whole class is one batch
        self.alpha_max = alpha_max      # max fraction of the parent->target gap a clone closes
        self.noise_std = noise_std      # std of the exploration noise
        self.callback = callback        # callback(event, **info); None = silent

        self._init_state()

    def _init_state(self):
        # CLONALG Variables
        self.population = []
        self.mem_vecs = None
        self.mem_cls = None
        self.history = {}               # class -> [(stage, accuracy), ...]
        self.trained_classes = []       # in the order they were trained
        self.last_metrics = None
        self.length = None
        self.knn = None
        self._comp_min = None
        self._comp_max = None
        self._stop = False

    # ===================================================================
    # CLONALG METHODS
    # ===================================================================

    # returns a list of 2d numpy arrays [Normalized]
    def create_population(self, cls_antigens=None):
        if cls_antigens is None:
            self.population = np.random.uniform(
                self._comp_min,
                self._comp_max,
                size=(self.population_size, self.length),
            )
            return

        pick = np.random.choice(
            len(cls_antigens),
            self.population_size,
            replace=len(cls_antigens) < self.population_size,
        )
        seeds = cls_antigens[pick] + np.random.normal(
            0,
            self.noise_std,
            size=(self.population_size, self.length),
        )
        self.population = np.clip(seeds, self._comp_min, self._comp_max)

    # Each antigen in the batch is owned by its nearest antibody.
    # Returns indices of antibodies that own >= 1 antigen, and each one's target
    # (mean of its antigens).
    def assign_targets(self, batch):
        owner = np.argmin(euclidean_distances(self.population, batch), axis=0)  # (B,)
        idx = np.unique(owner)
        targets = np.stack([batch[owner == i].mean(axis=0) for i in idx])
        return idx, targets

    # Directed clone + small noise: clone = parent + alpha * (target - parent) + noise
    def clone_and_mutate(self, sel_fit, sel_antibodies, sel_targets):
        counts = np.ceil(sel_fit / sel_fit.sum() * self.num_clones).astype(int)
        parents = np.repeat(sel_antibodies, counts, axis=0)
        targets = np.repeat(sel_targets, counts, axis=0)

        alpha = np.random.uniform(0, self.alpha_max, size=(len(parents), 1))
        clones = parents + alpha * (targets - parents)

        mask = np.random.random(clones.shape) < self.mutation_rate
        clones += mask * np.random.normal(0, self.noise_std, size=clones.shape)
        return np.clip(clones, self._comp_min, self._comp_max)

    # Class memory = the antibodies that own the most antigens of the class
    # (ties: closest to any antigen). Also returns the mean distance from each
    # antigen to its nearest memory cell, for monitoring.
    def build_memory(self, cls_antigens):
        d = euclidean_distances(self.population, cls_antigens)  # (P, N)
        counts = np.bincount(d.argmin(axis=0), minlength=len(self.population))
        order = np.lexsort((d.min(axis=1), -counts))             # primary: -counts
        mem = self.population[order[:self.memory_size]]
        nearest = euclidean_distances(cls_antigens, mem).min(axis=1).mean()
        return mem, nearest

    # Cull clustered antibodies
    def balance_population(self, pop):
        pop = pop[np.random.permutation(len(pop))]
        dist = euclidean_distances(pop)
        far = dist >= self.similarity_threshold
        keep, ok = [0], far[0].copy()

        for i in range(1, len(pop)):
            if ok[i]:
                keep.append(i)
                ok &= far[i]

        # floor: top up with the points farthest from what we've kept
        rest = np.setdiff1d(np.arange(len(pop)), keep)
        while len(keep) < self.memory_size and len(rest):
            j = rest[np.argmax(dist[np.ix_(rest, keep)].min(axis=1))]
            keep.append(j)
            rest = rest[rest != j]

        return pop[keep]

    # ===================================================================
    # EVENTS / CONTROL
    # ===================================================================
    def _emit(self, event, **info):
        if self.callback is not None:
            self.callback(event, **info)

    # Ask a running partial_fit to stop at the next batch boundary.
    def request_stop(self):
        self._stop = True

    def reset(self):
        """Forget everything: memory, history, trained classes, classifier."""
        self._init_state()

    # ===================================================================
    # API METHODS
    # ===================================================================
    def _init_space(self, length):
        if self.length is None:
            self.length = length
        elif self.length != length:
            raise ValueError(f"Antigen length {length} does not match the model ({self.length})")
        self._comp_min = 0
        self._comp_max = 1

    def _n_batches(self, n_antigens):
        bs = self.batch_size or n_antigens
        return int(np.ceil(n_antigens / bs))

    # Train on any classes in (antigens, labels) that the model has not seen yet.
    # Classes are trained one after another, in `class_order` (default: ascending
    # label), with an evaluation after each when eval_data is given.
    # Returns True if every pending class finished, False if stopped early.
    def partial_fit(self, antigens, labels, epochs=5, eval_data=None, class_order=None):
        antigens = self._flatten(antigens)
        labels = np.asarray(labels)
        self._init_space(antigens.shape[1])
        self._stop = False

        order = np.unique(labels) if class_order is None else class_order
        pending = [int(c) for c in order if int(c) not in self.trained_classes]

        total_steps = sum(
            epochs * self._n_batches(int((labels == c).sum())) for c in pending
        )
        step = 0
        self._emit('train_start', classes=pending, total_steps=total_steps)

        for k, class_ in enumerate(pending):
            cls_antigens = antigens[labels == class_]
            bs = self.batch_size or len(cls_antigens)
            n_batches = self._n_batches(len(cls_antigens))

            self.create_population(cls_antigens)
            self._emit(
                'class_start', cls=class_, k=k + 1, n_classes=len(pending),
                population=len(self.population), n_antigens=len(cls_antigens), batch_size=bs,
            )

            class_mem = None
            for epoch in range(epochs):
                pop_start, clones_added, pruned = len(self.population), 0, 0
                order_ = np.random.permutation(len(cls_antigens))

                for b, start in enumerate(range(0, len(order_), bs)):
                    if self._stop:
                        self._emit('stopped')
                        return False

                    batch = cls_antigens[order_[start:start + bs]]
                    idx, targets = self.assign_targets(batch)

                    # fitness: affinity of each owning antibody to its own target
                    fit = 1 / (1 + np.linalg.norm(self.population[idx] - targets, axis=1))
                    top = np.argsort(-fit)[:self.selection_size]
                    clones = self.clone_and_mutate(
                        fit[top],
                        self.population[idx[top]],
                        targets[top],
                    )

                    merged = np.concatenate([self.population, clones])
                    self.population = self.balance_population(merged)
                    clones_added += len(clones)
                    pruned += len(merged) - len(self.population)

                    step += 1
                    self._emit(
                        'batch_end', cls=class_, epoch=epoch + 1, epochs=epochs,
                        batch=b + 1, n_batches=n_batches, step=step, total_steps=total_steps,
                    )

                class_mem, nearest = self.build_memory(cls_antigens)
                self._emit(
                    'epoch_end', cls=class_, epoch=epoch + 1, epochs=epochs, nearest=nearest,
                    pop_start=pop_start, pop_end=len(self.population),
                    clones_added=clones_added, pruned=pruned,
                )

            # memory is only written once a class is fully trained, so stopping
            # mid-class leaves the model consistent
            self._add_memory(class_mem, class_)
            self.trained_classes.append(class_)
            self._emit('class_end', cls=class_, memory_size=len(self.mem_vecs))

            if eval_data is not None:
                self.evaluate_stage(eval_data)

        self._emit('train_end', trained_classes=list(self.trained_classes))
        return True

    # Train from scratch on everything passed in (script convenience).
    def fit(self, antigens, labels, epochs=5, eval_data=None):
        self.reset()
        return self.partial_fit(antigens, labels, epochs, eval_data)

    # Append one class's memory cells to the model memory
    def _add_memory(self, class_mem, class_):
        cls_labels = np.full(len(class_mem), class_)

        if self.mem_vecs is None:
            self.mem_vecs, self.mem_cls = class_mem, cls_labels
        else:
            self.mem_vecs = np.concatenate([self.mem_vecs, class_mem])
            self.mem_cls = np.concatenate([self.mem_cls, cls_labels])

        self._sync_classifier()

    # Rebuild the KNN so it always mirrors the current memory.
    def _sync_classifier(self):
        k = min(self.n_neighbors, len(self.mem_vecs))
        self.knn = KNeighborsClassifier(
            n_neighbors=k,
            weights=self.weights,
            metric=self.metric,
        )
        self.knn.fit(self.mem_vecs, self.mem_cls)

    @staticmethod
    def _flatten(data):
        data = np.asarray(data)

        if data.ndim > 3:
            raise Exception("Data is too fat.. we can't work with that")
        elif data.ndim == 3:
            data = data.reshape(data.shape[0], -1)

        return data

    # Score the test samples of every class trained so far
    # Returns a metrics dict [None below 2 classes] and records per-class
    # accuracy in self.history. Stage 1 is skipped on purpose.
    #
    #   stage, labels, accuracy, balanced_accuracy, new_class, new_class_recall,
    #   per_class {label: (accuracy, n)}, confusion (rows=true, cols=pred)
    def evaluate_stage(self, eval_data, seen=None):
        seen = list(self.trained_classes if seen is None else seen)
        if len(seen) < 2:
            return None

        x_test, y_test = eval_data
        y_test = np.asarray(y_test)
        test_mask = np.isin(y_test, seen)
        y_true = y_test[test_mask]
        y_pred = self.predict(np.asarray(x_test)[test_mask])

        stage = len(seen)
        new_class = seen[-1]
        per_class = {}
        for cls in seen:
            cls_mask = y_true == cls
            if cls_mask.sum() == 0:
                continue
            cls_acc = float(np.mean(y_pred[cls_mask] == cls))
            per_class[cls] = (cls_acc, int(cls_mask.sum()))
            self.history.setdefault(cls, []).append((stage, cls_acc))

        new_mask = y_true == new_class
        metrics = {
            'stage': stage,
            'labels': seen,
            'accuracy': float(np.mean(y_pred == y_true)),
            'balanced_accuracy': float(np.mean([a for a, _ in per_class.values()])),
            'new_class': new_class,
            'new_class_recall': float(np.mean(y_pred[new_mask] == new_class)) if new_mask.any() else 0.0,
            'per_class': per_class,
            'confusion': confusion_matrix(y_true, y_pred, labels=seen),
        }
        self.last_metrics = metrics
        self._emit('eval', metrics=metrics)
        return metrics

    # ===================================================================
    # METRIC ADAPTERS
    # ===================================================================
    def _require_trained(self):
        if self.knn is None:
            raise RuntimeError("The model has no memory yet - train it first")

    def score(self, x_test, y_test):
        self._require_trained()
        return self.knn.score(self._flatten(x_test), y_test)

    def predict(self, x_test):
        self._require_trained()
        return self.knn.predict(self._flatten(x_test))

    def predict_proba(self, x_test):
        self._require_trained()
        return self.knn.predict_proba(self._flatten(x_test))

    def classes_(self):
        self._require_trained()
        return self.knn.classes_

    # Everything the Test view needs about ONE sample (28x28 or flat):
    #   label, proba {label: p}, distances, neighbor_vecs, neighbor_cls, nearest,
    #   unknown (True if the nearest memory cell is farther than `unknown_dist`)
    def explain(self, x, k=None, unknown_dist=None):
        self._require_trained()
        x = np.asarray(x, dtype=float).reshape(1, -1)
        k = min(k or self.knn.n_neighbors, len(self.mem_vecs))
        dist, idx = self.knn.kneighbors(x, n_neighbors=k)
        proba = self.knn.predict_proba(x)[0]
        return {
            'label': int(self.knn.predict(x)[0]),
            'proba': {int(c): float(p) for c, p in zip(self.knn.classes_, proba)},
            'distances': dist[0],
            'neighbor_vecs': self.mem_vecs[idx[0]],
            'neighbor_cls': self.mem_cls[idx[0]],
            'nearest': float(dist[0][0]),
            'unknown': unknown_dist is not None and float(dist[0][0]) > unknown_dist,
        }

    # ===================================================================
    # PLOTS (draw onto a given Axes/Figure; theming is the caller's job)
    # ===================================================================
    def plot_history(self, ax=None, class_names=None):
        import matplotlib.pyplot as plt

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 5))

        for cls, points in sorted(self.history.items()):
            stages, accs = zip(*points)
            label = class_names[cls] if class_names else f'class {cls}'
            ax.plot(stages, accs, marker='o', label=label)

        ax.set_xlabel('Classes seen (stage)')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.set_title('Per-class accuracy over training stages')

        if standalone:
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.figure.tight_layout()
            plt.show()
        return ax.figure

    def plot_memory(self, fig=None, per_class=10, img_shape=(28, 28), class_names=None):
        import matplotlib.pyplot as plt

        classes = np.unique(self.mem_cls)
        n = min(per_class, self.memory_size)

        if fig is None:
            fig = plt.figure(figsize=(n * 1.2, len(classes) * 1.2))
            standalone = True
        else:
            fig.clear()
            standalone = False
        axes = fig.subplots(len(classes), n, squeeze=False)

        for r, cls in enumerate(classes):
            cells = self.mem_vecs[self.mem_cls == cls][:n]

            for c in range(n):
                ax = axes[r, c]
                ax.axis('off')

                if c < len(cells):
                    ax.imshow(
                        cells[c].reshape(img_shape),
                        cmap='gray',
                        vmin=0,
                        vmax=1,
                    )

                if c == 0:
                    title = class_names[cls] if class_names else f'class {cls}'
                    ax.set_title(title, fontsize=8, loc='left')

        fig.suptitle('Memory cells per class')
        fig.tight_layout()
        if standalone:
            plt.show()
        return fig
