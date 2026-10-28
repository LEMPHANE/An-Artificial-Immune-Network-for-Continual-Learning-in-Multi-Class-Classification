"""
Plot theme + Qt-embedded matplotlib canvases for the GUI.

    apply_theme(ax)        style any Axes in the app theme
    class_style(label)     colour / marker / hollow for a class label (one scheme everywhere)

    HistoryCanvas          per-class accuracy over training stages      (footer, always visible)
    TrainCurveCanvas       mean antigen -> nearest memory distance       (Training view)
    MemoryGridCanvas       grid of memory-cell images per class          (Memory view, tab 2)
    PCACanvas              interactive PCA of the memory pool            (Memory view, tab 1)
    ConfusionCanvas        row-normalised confusion matrix               (Metrics view)

All canvases are plain QWidgets: drop one into a layout. They must only be touched from
the GUI thread (connect worker signals to slots, never call them from the worker).

--- colour scheme ------------------------------------------------------------------------
Hues are the validated categorical palette (dark steps), run through
validate_palette.js against the plot surface #17191D:
  - 8 slots, adjacent pairs (lines + legend): PASS (worst CVD dE 8.4, normal-vision 19.3)
  - all-pairs (scatter): only the first 3 slots clear the floors; all 8 FAIL.
With up to 20 classes hue alone cannot identify a class in the scatter, so identity is
a composite of three channels:
    hue     slot (index within the dataset) % 8
    marker  circle = MNIST, square = Fashion-MNIST
    fill    filled for index 0-7, hollow ring for index 8-9 (re-uses hue slots 0-1)
Every (marker, hue, fill) combination is unique. Hover names the class, legend entries
toggle / isolate classes, so nothing relies on colour alone.
"""
import warnings

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.ticker import MaxNLocator, PercentFormatter
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QSizePolicy
from sklearn.decomposition import PCA

from data.data_store import FASHION_OFFSET, class_name

# A canvas can be painted once before the layout has given it a real size; matplotlib then skips
# constrained layout and warns. It lays out properly on the next resize, so the warning is only noise.
warnings.filterwarnings("ignore", message="constrained_layout not applied because axes sizes collapsed")

# -------------------------------------------------------------------
# Theme tokens (match ui/styles.qss)
# -------------------------------------------------------------------
PANEL = "#1B1D21"      # figure margin (same as the QSS panel colour)
SURFACE = "#17191D"    # plotting area - darker than the margin so the plot reads as inset
BORDER = "#373A40"     # figure outline + axis spines
GRID = "#2B2D32"       # hairline grid, one step off the surface
INK = "#F5F6F8"        # primary text
INK_2 = "#A5A9B2"      # secondary text
MUTED = "#8B8F98"      # tick labels
ACCENT = "#C84628"
ACCENT_HOT = "#EB5B3B"

# Categorical hues, dark-mode steps (fixed order, assigned by class - never by rank)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

THEME_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
    "font.size": 8,
    "text.color": INK_2,
    "figure.facecolor": PANEL,
    "figure.edgecolor": BORDER,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
}

CM_CMAP = LinearSegmentedColormap.from_list("jarvis_seq", ["#1E2024", "#7A3322", ACCENT_HOT])


def apply_theme(ax, grid="y"):
    """Style an Axes (and its figure) in the app theme. grid: 'x' | 'y' | 'both' | None."""
    fig = ax.figure
    fig.set_facecolor(PANEL)
    fig.set_edgecolor(BORDER)
    fig.patch.set_linewidth(2)
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
        spine.set_visible(side in ("left", "bottom"))
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


# -------------------------------------------------------------------
# Class identity (colour + marker + fill) and names
# -------------------------------------------------------------------
def class_style(label):
    label = int(label)
    fashion = label >= FASHION_OFFSET
    idx = label - FASHION_OFFSET if fashion else label
    return {
        "color": SERIES[idx % len(SERIES)],
        "marker": "s" if fashion else "o",
        "hollow": idx >= len(SERIES),
    }


def marker_kwargs(style, ms=6.0):
    """Line2D marker kwargs: filled marker with a surface ring, or a hollow coloured ring."""
    c = style["color"]
    size = ms * 0.9 if style["marker"] == "s" else ms   # squares look bigger at equal size
    if style["hollow"]:
        return dict(marker=style["marker"], markersize=size, markerfacecolor=SURFACE,
                    markeredgecolor=c, markeredgewidth=1.6)
    return dict(marker=style["marker"], markersize=size, markerfacecolor=c,
                markeredgecolor=SURFACE, markeredgewidth=1.2)


def short_name(label):
    """'MNIST: 5' -> '5', 'Fashion: Sneaker' -> 'Sneaker' (for tick labels)."""
    name = class_name(label)
    return name.split(": ", 1)[1] if ": " in name else name


def _legend_handle(label, line=True):
    style = class_style(label)
    return Line2D([], [], linestyle="-" if line else "", linewidth=1.5, color=style["color"],
                  **marker_kwargs(style, ms=6.5))


def _nearest(canvas, ax, pts, event, radius=24):
    """Index of the point in `pts` (data coords) within `radius` logical px of the mouse, else None."""
    if pts is None or len(pts) == 0 or event.inaxes is not ax:
        return None
    ratio = getattr(canvas, "device_pixel_ratio", 1) or 1
    d = np.hypot(*(ax.transData.transform(pts) - [event.x, event.y]).T)
    i = int(d.argmin())
    return i if d[i] <= radius * ratio else None


def _drop(artist):
    if artist is not None:
        try:
            artist.remove()
        except (ValueError, NotImplementedError, AttributeError):
            pass


# -------------------------------------------------------------------
# Base canvas
# -------------------------------------------------------------------
class ThemedCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, size=(5.0, 3.0), layout="constrained"):
        fig = Figure(figsize=size, facecolor=PANEL, edgecolor=BORDER, linewidth=2, layout=layout)
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = None

    def _new_axes(self, grid="y"):
        self.figure.clear()
        with mpl.rc_context(THEME_RC):
            self.ax = self.figure.add_subplot(111)
        apply_theme(self.ax, grid)
        return self.ax

    def _empty(self, text):
        ax = self._new_axes(grid=None)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                color=MUTED, fontsize=9)
        self.draw_idle()


# -------------------------------------------------------------------
# Line charts: per-class history, training curve
# -------------------------------------------------------------------
class LineChartCanvas(ThemedCanvas):
    """One line per class, class colours, hover tooltip on the nearest point."""

    TITLE = ""
    XLABEL = ""
    YLABEL = ""
    EMPTY_TEXT = "Nothing to show yet"
    Y_PERCENT = False
    Y_LIM = None   # (lo, hi) or None for auto

    def __init__(self, parent=None, size=(6.0, 2.6)):
        super().__init__(parent, size)
        self._series = {}      # label -> [(x, y), ...]
        self._tip = None
        self._tip_idx = None
        self._pts = None
        self._info = []
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("figure_leave_event", lambda e: self._hide_tip())
        self._render()

    # ---- data ----------------------------------------------------------
    def set_series(self, series):
        self._series = {int(c): sorted(p) for c, p in series.items() if len(p)}
        self._render()

    def clear(self):
        self._series = {}
        self._render()

    def _tip_text(self, label, x, y):
        return f"{class_name(label)}\n{x:g}: {y:.3f}"

    # ---- drawing -------------------------------------------------------
    def _render(self):
        self._tip, self._tip_idx = None, None
        if not self._series:
            self._pts, self._info = None, []
            self._empty(self.EMPTY_TEXT)
            return

        ax = self._new_axes(grid="y")
        pts, info = [], []
        for label in sorted(self._series):
            xs, ys = zip(*self._series[label])
            style = class_style(label)
            ax.plot(xs, ys, color=style["color"], linewidth=1.5, solid_capstyle="round",
                    zorder=3, **marker_kwargs(style))
            pts += list(zip(xs, ys))
            info += [(label, x, y) for x, y in zip(xs, ys)]
        self._pts, self._info = np.array(pts, float), info

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        x_lo, x_hi = self._pts[:, 0].min(), self._pts[:, 0].max()
        ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
        if self.Y_LIM:
            ax.set_ylim(*self.Y_LIM)
        else:
            ax.set_ylim(0, max(self._pts[:, 1].max() * 1.1, 1e-6))
        if self.Y_PERCENT:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(self.TITLE, loc="left")
        ax.set_xlabel(self.XLABEL)
        ax.set_ylabel(self.YLABEL)

        labels = sorted(self._series)
        if len(labels) >= 2:     # legend is always present for >= 2 series
            self.figure.legend(
                [_legend_handle(c) for c in labels], [class_name(c) for c in labels],
                loc="outside right center", fontsize=7, ncol=1 if len(labels) <= 10 else 2,
                handlelength=2.2, columnspacing=1.2, labelspacing=0.4,
                frameon=False, labelcolor=INK_2,
            )
        if len(labels) <= 4:     # + direct labels at the line ends (skipped where they would collide)
            self._direct_labels(ax, labels)
        self.draw_idle()

    def _direct_labels(self, ax, labels):
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ends = sorted(((self._series[c][-1][1], self._series[c][-1][0], c) for c in labels), reverse=True)
        last_y = None
        for y, x, c in ends:
            if last_y is not None and last_y - y < 0.07 * span:
                continue
            ax.annotate(short_name(c), (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color=INK_2, fontsize=7.5, annotation_clip=False)
            last_y = y

    # ---- hover ---------------------------------------------------------
    def _on_move(self, event):
        i = _nearest(self, self.ax, self._pts, event)
        if i is None:
            self._hide_tip()
        elif i != self._tip_idx:
            self._show_tip(i)

    def _show_tip(self, i):
        _drop(self._tip)
        label, x, y = self._info[i]
        ax = self.ax
        right = ax.transData.transform((x, y))[0] > ax.bbox.x0 + 0.65 * ax.bbox.width
        self._tip = ax.annotate(
            self._tip_text(label, x, y), (x, y),
            xytext=(-10 if right else 10, 10), textcoords="offset points",
            ha="right" if right else "left", va="bottom", color=INK, fontsize=7.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=BORDER, lw=1),
        )
        self._tip_idx = i
        self.draw_idle()

    def _hide_tip(self):
        if self._tip is not None:
            _drop(self._tip)
            self._tip, self._tip_idx = None, None
            self.draw_idle()


class HistoryCanvas(LineChartCanvas):
    """Per-class accuracy as classes are introduced (CLONALG.history; starts at stage 2)."""
    TITLE = "Per-class accuracy over training stages"
    XLABEL = "Classes seen (stage)"
    YLABEL = "Accuracy"
    EMPTY_TEXT = "Train at least two classes to see per-class accuracy"
    Y_PERCENT = True
    Y_LIM = (0, 1.05)

    def update_history(self, history):
        self.set_series(history)

    def _tip_text(self, label, x, y):
        return f"{class_name(label)}\nstage {x:g}: {y:.1%}"


class TrainCurveCanvas(LineChartCanvas):
    """Mean distance from each antigen to its nearest memory cell, per epoch (lower = better fit)."""
    TITLE = "Antigen to nearest memory cell"
    XLABEL = "Epoch"
    YLABEL = "Mean distance"
    EMPTY_TEXT = "Start training to see the fit improve"

    def add_point(self, label, epoch, value):
        self._series.setdefault(int(label), []).append((int(epoch), float(value)))
        self.set_series(self._series)

    def reset(self):
        self.clear()

    def _tip_text(self, label, x, y):
        return f"{class_name(label)}\nepoch {x:g}: {y:.3f}"


# -------------------------------------------------------------------
# Memory cell grid
# -------------------------------------------------------------------
class MemoryGridCanvas(ThemedCanvas):
    """Memory cells per class, as images. Put it in a QScrollArea: it asks for ~58 px per class."""

    def __init__(self, parent=None):
        super().__init__(parent, size=(6.0, 4.0), layout=None)   # model.plot_memory uses tight_layout
        self._empty("No memory cells yet - train a class")

    def clear(self):
        self.setMinimumHeight(0)
        self._empty("No memory cells yet - train a class")

    def update_model(self, model, per_class=10, img_shape=(28, 28)):
        if model.mem_vecs is None:
            self._empty("No memory cells yet - train a class")
            return
        classes = [int(c) for c in np.unique(model.mem_cls)]
        n = min(per_class, model.memory_size)
        height = max(160, 58 * len(classes) + 40)
        self.setMinimumHeight(height)

        fig = self.figure
        fig.clear()
        fig.set_facecolor(PANEL)
        with mpl.rc_context(THEME_RC):
            axes = fig.subplots(
                len(classes), n, squeeze=False,
                gridspec_kw=dict(left=0.2, right=0.995, top=1 - 30 / height, bottom=6 / height,
                                 wspace=0.08, hspace=0.08),
            )
        fig.text(0.012, 1 - 10 / height, "Memory cells per class", color=INK, fontsize=9,
                 ha="left", va="top")

        for r, cls in enumerate(classes):
            cells = model.mem_vecs[model.mem_cls == cls][:n]
            for c in range(n):
                ax = axes[r, c]
                ax.axis("off")
                if c < len(cells):
                    ax.imshow(np.clip(cells[c].reshape(img_shape), 0, 1), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
                if c == 0:
                    ax.text(-0.1, 0.5, class_name(cls), transform=ax.transAxes, ha="right",
                            va="center", fontsize=7.5, color=INK)
        self.draw_idle()


# -------------------------------------------------------------------
# Confusion matrix
# -------------------------------------------------------------------
class ConfusionCanvas(ThemedCanvas):
    """Row-normalised confusion matrix (share of each true class), counts on cells when <= 10 classes."""

    def __init__(self, parent=None):
        super().__init__(parent, size=(5.0, 4.4))
        self._cm = None
        self._labels = []
        self._tip = None
        self._tip_cell = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("figure_leave_event", lambda e: self._hide_tip())
        self._empty("Train at least two classes to see the confusion matrix")

    def update_confusion(self, cm, labels):
        cm = np.asarray(cm)
        self._cm, self._labels = cm, [int(l) for l in labels]
        self._tip, self._tip_cell = None, None
        n = len(self._labels)
        if n == 0 or cm.size == 0:
            self._empty("Train at least two classes to see the confusion matrix")
            return

        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax = self._new_axes(grid=None)
        im = ax.imshow(norm, cmap=CM_CMAP, vmin=0, vmax=1, interpolation="nearest")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        tick_names = [short_name(l) for l in self._labels]
        rotate = any(len(t) > 2 for t in tick_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_names, rotation=45 if rotate else 0, ha="right" if rotate else "center")
        ax.set_yticklabels(tick_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix", loc="left")

        # 2px surface gap between cells
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)
        ax.tick_params(which="minor", length=0)

        if n <= 10:   # counts only on non-empty cells, only while they fit
            for i in range(n):
                for j in range(n):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                                color=INK if norm[i, j] > 0.5 else INK_2)

        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Share of true class", color=INK_2, fontsize=7.5)
        cbar.ax.tick_params(colors=MUTED, labelsize=7, length=2)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        cbar.outline.set_edgecolor(BORDER)
        self.draw_idle()

    # ---- hover ---------------------------------------------------------
    def _on_move(self, event):
        if self._cm is None or event.inaxes is not self.ax or event.xdata is None:
            self._hide_tip()
            return
        j, i = int(round(event.xdata)), int(round(event.ydata))
        n = len(self._labels)
        if not (0 <= i < n and 0 <= j < n):
            self._hide_tip()
            return
        if (i, j) == self._tip_cell:
            return
        _drop(self._tip)
        total = self._cm[i].sum()
        count = int(self._cm[i, j])
        share = count / total if total else 0.0
        self._tip = self.ax.annotate(
            f"true {class_name(self._labels[i])}\npredicted {class_name(self._labels[j])}\n"
            f"{count} of {int(total)} ({share:.1%})",
            (j, i), xytext=(10, -10), textcoords="offset points", ha="left", va="top",
            color=INK, fontsize=7.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=BORDER, lw=1),
        )
        self._tip_cell = (i, j)
        self.draw_idle()

    def _hide_tip(self):
        if self._tip is not None:
            _drop(self._tip)
            self._tip, self._tip_cell = None, None
            self.draw_idle()


# -------------------------------------------------------------------
# Memory pool PCA
# -------------------------------------------------------------------
class PCACanvas(ThemedCanvas):
    """
    Memory cells projected to 2-D with PCA (colour/marker per class, see class_style).

      update_data(mem_vecs, mem_cls, antigens, antigen_cls)
          Pass EVERY loaded training antigen (trained or not): the PCA is fitted on them, so
          the axes only move when classes are loaded, not when a class finishes training.
          Antigens of trained classes are drawn as a faint cloud behind the memory cells.
      set_probe(vec, neighbors)   a drawn sample as a star (+ lines to its nearest memory cells)

    Interaction: hover a memory cell -> its image and class; click it to pin the tip, click
    empty space to unpin; click a legend entry to hide/show that class, double-click to isolate
    it, double-click the plot to show everything again.
    """

    def __init__(self, parent=None, img_shape=(28, 28)):
        super().__init__(parent, size=(6.0, 4.2))
        self.img_shape = img_shape
        self._pca = None
        self._mem = self._mem_cls = self._mem2 = None
        self._ant2 = self._ant_cls = None
        self._var = (0.0, 0.0)
        self._hidden = set()
        self._probe = None             # (probe_xy, neighbour_xy or None)
        self._pts = None               # visible memory cells in data coords
        self._pts_idx = None           # their row in self._mem
        self._tip = self._tip_idx = None
        self._pin = self._pin_idx = None
        self._leg_map = {}
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("pick_event", self._on_pick)
        self.mpl_connect("figure_leave_event", lambda e: self._hide_tip())
        self._empty("No memory cells yet - train a class")

    # ---- data ----------------------------------------------------------
    def update_data(self, mem_vecs, mem_cls, antigens=None, antigen_cls=None):
        if mem_vecs is None or len(mem_vecs) == 0:
            self._mem = None
            self._probe = None
            self._empty("No memory cells yet - train a class")
            return
        mem = np.asarray(mem_vecs, float).reshape(len(mem_vecs), -1)
        mem_cls = np.asarray(mem_cls, int)

        ant = ant_cls = None
        if antigens is not None and len(antigens) >= 3:
            ant = np.asarray(antigens, float).reshape(len(antigens), -1)
            ant_cls = np.asarray(antigen_cls, int)

        self._pca = PCA(n_components=2, random_state=0).fit(ant if ant is not None else mem)
        self._var = tuple(self._pca.explained_variance_ratio_)
        self._mem, self._mem_cls, self._mem2 = mem, mem_cls, self._pca.transform(mem)

        self._ant2 = self._ant_cls = None
        if ant is not None:          # context cloud: trained classes only, capped per class
            rng = np.random.default_rng(0)
            keep = []
            for c in np.unique(mem_cls):
                idx = np.flatnonzero(ant_cls == c)
                keep.append(rng.choice(idx, min(len(idx), 150), replace=False) if len(idx) else idx)
            keep = np.concatenate(keep) if keep else np.array([], int)
            if len(keep):
                self._ant2, self._ant_cls = self._pca.transform(ant[keep]), ant_cls[keep]

        self._hidden &= set(int(c) for c in np.unique(mem_cls))
        self._probe = None           # the old probe was projected with the old PCA
        self._redraw()

    def set_probe(self, vec, neighbors=None, neighbor_cls=None):
        """Show a drawn sample as a star.

        vec: flat or image. neighbors: (k, D) nearest memory cells (lines are drawn to them);
        neighbor_cls: their labels, so lines to hidden classes are skipped.
        """
        if vec is None or self._pca is None:
            self._probe = None
        else:
            p = self._pca.transform(np.asarray(vec, float).reshape(1, -1))[0]
            nb = None if neighbors is None else self._pca.transform(
                np.asarray(neighbors, float).reshape(len(neighbors), -1))
            nb_cls = None if neighbor_cls is None else [int(c) for c in neighbor_cls]
            self._probe = (p, nb, nb_cls)
        if self._mem is not None:
            self._redraw()

    def clear_probe(self):
        self.set_probe(None)

    # ---- drawing -------------------------------------------------------
    def _redraw(self):
        self._tip = self._tip_idx = self._pin = self._pin_idx = None
        ax = self._new_axes(grid="both")
        classes = [int(c) for c in np.unique(self._mem_cls)]
        vis = ~np.isin(self._mem_cls, list(self._hidden))
        self._pts, self._pts_idx = self._mem2[vis], np.flatnonzero(vis)

        for c in classes:
            if c in self._hidden:
                continue
            st = class_style(c)
            if self._ant2 is not None:
                a = self._ant2[self._ant_cls == c]
                if len(a):
                    ax.scatter(a[:, 0], a[:, 1], s=6, c=st["color"], marker=st["marker"],
                               alpha=0.16, linewidths=0, zorder=2)
            m = self._mem2[self._mem_cls == c]
            hollow = st["hollow"]
            ax.scatter(m[:, 0], m[:, 1], s=62, marker=st["marker"],
                       facecolors=SURFACE if hollow else st["color"],
                       edgecolors=st["color"] if hollow else SURFACE,
                       linewidths=1.6 if hollow else 1.2, zorder=4)

        if self._probe is not None:
            p, nb, nb_cls = self._probe
            if nb is not None:
                for k, q in enumerate(nb):
                    if nb_cls is not None and nb_cls[k] in self._hidden:
                        continue
                    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK_2, linewidth=1, alpha=0.7, zorder=5)
            ax.scatter([p[0]], [p[1]], s=190, marker="*", facecolors=INK, edgecolors=SURFACE,
                       linewidths=1.2, zorder=6)
            ax.annotate("your drawing", p, xytext=(8, 8), textcoords="offset points",
                        color=INK, fontsize=7.5, zorder=7)

        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.08)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.set_xlabel(f"PC1 ({self._var[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({self._var[1]:.0%})")
        ax.set_title("Memory pool (PCA)", loc="left")
        self._build_legend(classes)
        self.draw_idle()

    def _build_legend(self, classes):
        self._leg_map = {}
        handles = [_legend_handle(c, line=False) for c in classes]
        leg = self.figure.legend(
            handles, [class_name(c) for c in classes], loc="outside right center", fontsize=7,
            ncol=1 if len(classes) <= 10 else 2, columnspacing=1.2, labelspacing=0.45,
            frameon=False, labelcolor=INK_2,
        )
        entries = getattr(leg, "legend_handles", None) or leg.legendHandles
        for c, h, t in zip(classes, entries, leg.get_texts()):
            h.set_picker(8)
            t.set_picker(True)
            if c in self._hidden:
                h.set_alpha(0.3)
                t.set_alpha(0.4)
            self._leg_map[h] = c
            self._leg_map[t] = c

    # ---- legend toggling -----------------------------------------------
    def _on_pick(self, event):
        c = self._leg_map.get(event.artist)
        if c is None:
            return
        all_classes = set(int(x) for x in np.unique(self._mem_cls))
        if getattr(event.mouseevent, "dblclick", False):
            self._hidden = all_classes - {c}            # isolate
        elif c in self._hidden:
            self._hidden.discard(c)
        else:
            self._hidden.add(c)
        QTimer.singleShot(0, self._redraw)              # never rebuild the legend inside its own pick

    # ---- hover / pin ---------------------------------------------------
    def _make_tip(self, i, pinned=False):
        row = self._pts_idx[i]
        c = int(self._mem_cls[row])
        x, y = self._pts[i]
        img = np.clip(self._mem[row].reshape(self.img_shape), 0, 1)
        rgb = np.repeat(img[..., None], 3, axis=2)
        zoom = 44.0 / max(self.img_shape)
        box = VPacker(
            children=[
                OffsetImage(rgb, zoom=zoom, interpolation="nearest"),
                TextArea(class_name(c), textprops=dict(color=INK, fontsize=7.5)),
            ],
            pad=3, sep=3, align="center",
        )
        ax = self.ax
        right = ax.transData.transform((x, y))[0] > ax.bbox.x0 + 0.6 * ax.bbox.width
        tip = AnnotationBbox(
            box, (x, y), xybox=(-16 if right else 16, 16), xycoords="data",
            boxcoords="offset points", box_alignment=(1, 0) if right else (0, 0),
            bboxprops=dict(fc=PANEL, ec=ACCENT_HOT if pinned else BORDER, lw=1), zorder=10,
            arrowprops=dict(arrowstyle="-", color=BORDER, lw=0.8),
        )
        ax.add_artist(tip)
        return tip

    def _on_move(self, event):
        i = _nearest(self, self.ax, self._pts, event) if self._mem is not None else None
        if i is None:
            self._hide_tip()
        elif i != self._tip_idx and i != self._pin_idx:
            _drop(self._tip)
            self._tip, self._tip_idx = self._make_tip(i), i
            self.draw_idle()

    def _hide_tip(self):
        if self._tip is not None:
            _drop(self._tip)
            self._tip, self._tip_idx = None, None
            self.draw_idle()

    def _on_click(self, event):
        if self._mem is None or event.inaxes is not self.ax:
            return
        if getattr(event, "dblclick", False):           # show every class again
            self._hidden.clear()
            QTimer.singleShot(0, self._redraw)
            return
        i = _nearest(self, self.ax, self._pts, event)
        _drop(self._pin)
        self._pin, self._pin_idx = None, None
        if i is not None:
            _drop(self._tip)
            self._tip, self._tip_idx = None, None
            self._pin, self._pin_idx = self._make_tip(i, pinned=True), i
        self.draw_idle()


# -------------------------------------------------------------------
# Prediction result (Test view)
# -------------------------------------------------------------------
class PredictionCanvas(ThemedCanvas):
    """Result of classifying one drawing: predicted class, the vote among the nearest memory
    cells, and those cells as images with their distances (CLONALG.explain() output)."""

    def __init__(self, parent=None, img_shape=(28, 28)):
        super().__init__(parent, size=(3.6, 4.4), layout=None)
        self.img_shape = img_shape
        self.clear()

    def clear(self):
        self._empty("Draw a digit and press OK\nto classify it")

    def show_result(self, info):
        fig = self.figure
        fig.clear()
        fig.set_facecolor(PANEL)
        cells, dists, cls = info["neighbor_vecs"], info["distances"], info["neighbor_cls"]
        k = len(cells)
        votes = sorted(((p, c) for c, p in info["proba"].items() if p > 0), reverse=True)[:5]

        with mpl.rc_context(THEME_RC):
            if info["unknown"]:     # flagged, but the KNN guess stays visible next to it
                head = "Unknown"
                sub = f"Best guess: {class_name(info['label'])} ({info['nearest']:.2f} from the nearest cell)"
            else:
                head, sub = class_name(info["label"]), f"Nearest memory cell: {info['nearest']:.2f} away"
            fig.text(0.06, 0.96, head, color=INK, fontsize=14, ha="left", va="top")
            fig.text(0.06, 0.885, sub, color=INK_2, fontsize=7.5, ha="left", va="top")

            # vote bars: the % is written on each bar, so no x axis / grid is needed
            ax = fig.add_axes([0.42, 0.50, 0.50, 0.25])
            apply_theme(ax, grid=None)
            ax.spines["bottom"].set_visible(False)
            n = len(votes)
            ax.barh(range(n), [p for p, _ in votes], height=0.5,
                    color=[class_style(c)["color"] for _, c in votes])
            ax.set_yticks(range(n))
            ax.set_yticklabels([class_name(c) for _, c in votes])
            ax.tick_params(axis="y", colors=INK_2, labelsize=7, length=0)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.25)
            ax.set_xticks([])
            fig.text(0.06, 0.80, "Vote among the nearest cells", color=INK, fontsize=8, ha="left", va="top")
            for i, (p, _) in enumerate(votes):
                ax.text(p + 0.03, i, f"{p:.0%}", va="center", color=INK_2, fontsize=7.5)

            fig.text(0.06, 0.44, "Nearest memory cells", color=MUTED, fontsize=7.5, ha="left", va="top")
            w = 0.86 / k - 0.03
            for j in range(k):
                axi = fig.add_axes([0.07 + j * (w + 0.03), 0.07, w, 0.29])
                axi.imshow(np.clip(np.asarray(cells[j]).reshape(self.img_shape), 0, 1), cmap="gray",
                           vmin=0, vmax=1, interpolation="nearest")
                axi.axis("off")
                axi.set_title(f"{short_name(cls[j])}\n{dists[j]:.2f}", fontsize=7.5, color=INK_2)
        self.draw_idle()
