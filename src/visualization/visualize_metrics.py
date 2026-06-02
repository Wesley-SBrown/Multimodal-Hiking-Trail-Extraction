"""
src/visualization/visualize_metrics.py

Plots training diagnostics for the multimodal trail segmentation model:
  - Training loss & validation loss (shared axes)
  - IoU over epochs
  - Dice score over epochs
  - Precision / Recall over epochs

Usage
-----
Call `plot_training_metrics(history)` where `history` is a dict of lists, e.g.:

    history = {
        "train_loss":  [0.85, 0.71, ...],
        "val_loss":    [0.90, 0.74, ...],
        "iou":         [0.32, 0.41, ...],
        "dice":        [0.48, 0.57, ...],
        "precision":   [0.61, 0.68, ...],
        "recall":      [0.55, 0.63, ...],
    }

All lists must have the same length (one value per epoch).

The function also accepts an optional `save_path` string; if provided the
figure is written to that path in addition to being displayed.
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── aesthetic constants ────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_MAIN = "#e6edf3"
TEXT_DIM  = "#8b949e"
ACCENT_A  = "#58a6ff"   # train loss / IoU
ACCENT_B  = "#f85149"   # val loss
ACCENT_C  = "#3fb950"   # dice
ACCENT_D  = "#d2a8ff"   # precision
ACCENT_E  = "#ffa657"   # recall

FONT_TITLE = {"fontsize": 10, "fontweight": "bold", "color": TEXT_MAIN, "pad": 8}
FONT_LABEL = {"fontsize": 8,  "color": TEXT_DIM}

LINE_W   = 1.8
MARKER_S = 4


# ── helpers ────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str, ylabel: str, epochs: np.ndarray):
    """Apply shared dark-theme styling to a subplot axis."""
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_DIM, labelsize=7.5)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.set_title(title, **FONT_TITLE)
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel(ylabel, **FONT_LABEL)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(which="major", color=GRID_COL, linewidth=0.6, linestyle="--")
    ax.grid(which="minor", color=GRID_COL, linewidth=0.3, linestyle=":")


def _smooth(values: list, window: int = 3) -> np.ndarray:
    """Very light moving-average smoothing for dashed trend lines."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _best_epoch_marker(ax, epochs, values, color, label_prefix="best"):
    """Drop a vertical hairline + annotation at the best (min/max) epoch."""
    arr = np.array(values)
    if "loss" in label_prefix.lower():
        idx = int(np.argmin(arr))
        note = f"min {arr[idx]:.4f}"
    else:
        idx = int(np.argmax(arr))
        note = f"max {arr[idx]:.4f}"
    ax.axvline(epochs[idx], color=color, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.annotate(
        f" {note}",
        xy=(epochs[idx], arr[idx]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=6.5,
        color=color,
        alpha=0.85,
    )


def _legend(ax):
    """Consistent legend style across all panels."""
    ax.legend(
        fontsize=7.5,
        framealpha=0.15,
        facecolor=PANEL_BG,
        edgecolor=GRID_COL,
        labelcolor=TEXT_DIM,
    )


# ── main public function ───────────────────────────────────────────────────────

def plot_training_metrics(
    history: dict,
    save_path: str | None = None,
    region: str = "",
    smooth_window: int = 3,
) -> plt.Figure:
    """
    Parameters
    ----------
    history      : dict with keys train_loss, val_loss, iou, dice,
                   precision, recall (each a list of floats, one per epoch).
    save_path    : optional file path (.png / .pdf) to save the figure.
    region       : optional region name shown in the super-title.
    smooth_window: moving-average window for light trend overlay lines.

    Returns
    -------
    matplotlib Figure object.
    """
    required = {"train_loss", "val_loss", "iou", "dice", "precision", "recall"}
    missing  = required - set(history.keys())
    if missing:
        raise ValueError(f"history dict is missing keys: {missing}")

    n_epochs = len(history["train_loss"])
    if not all(len(history[k]) == n_epochs for k in required):
        raise ValueError("All history lists must have the same length.")

    epochs = np.arange(1, n_epochs + 1)

    # ── figure setup ──────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "text.color":        TEXT_MAIN,
        "axes.facecolor":    PANEL_BG,
        "savefig.facecolor": DARK_BG,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    fig.patch.set_facecolor(DARK_BG)

    gs = GridSpec(
        2, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.88,  bottom=0.08,
        hspace=0.42, wspace=0.32,
    )

    ax_loss = fig.add_subplot(gs[0, :2])  # wide: spans 2 cols
    ax_pr   = fig.add_subplot(gs[0, 2])   # precision / recall
    ax_iou  = fig.add_subplot(gs[1, 0])
    ax_dice = fig.add_subplot(gs[1, 1])
    ax_f1   = fig.add_subplot(gs[1, 2])   # derived F1

    # ── 1. Loss panel ─────────────────────────────────────────────────────────
    _style_ax(ax_loss, "Training vs Validation Loss", "Loss", epochs)

    ax_loss.plot(epochs, history["train_loss"], color=ACCENT_A, lw=LINE_W,
                 marker="o", markersize=MARKER_S, label="Train loss", zorder=3)
    ax_loss.plot(epochs, history["val_loss"],   color=ACCENT_B, lw=LINE_W,
                 marker="s", markersize=MARKER_S, label="Val loss",   zorder=3)

    if n_epochs >= smooth_window:
        ax_loss.plot(epochs, _smooth(history["train_loss"], smooth_window),
                     color=ACCENT_A, lw=0.8, alpha=0.4, linestyle="--", zorder=2)
        ax_loss.plot(epochs, _smooth(history["val_loss"],   smooth_window),
                     color=ACCENT_B, lw=0.8, alpha=0.4, linestyle="--", zorder=2)

    _best_epoch_marker(ax_loss, epochs, history["val_loss"], ACCENT_B, "loss")

    train_arr = np.array(history["train_loss"])
    val_arr   = np.array(history["val_loss"])
    ax_loss.fill_between(
        epochs, train_arr, val_arr,
        where=(val_arr > train_arr),
        color=ACCENT_B, alpha=0.07, label="Overfitting gap",
    )

    _legend(ax_loss)

    # ── 2. Precision / Recall panel ───────────────────────────────────────────
    _style_ax(ax_pr, "Precision & Recall", "Score", epochs)
    ax_pr.set_ylim(0, 1.05)

    ax_pr.plot(epochs, history["precision"], color=ACCENT_D, lw=LINE_W,
               marker="^", markersize=MARKER_S, label="Precision")
    ax_pr.plot(epochs, history["recall"],    color=ACCENT_E, lw=LINE_W,
               marker="v", markersize=MARKER_S, label="Recall")
    ax_pr.fill_between(epochs, history["precision"], history["recall"],
                       alpha=0.08, color=ACCENT_D)

    _legend(ax_pr)

    # ── 3. IoU panel ──────────────────────────────────────────────────────────
    _style_ax(ax_iou, "Intersection over Union (IoU)", "IoU", epochs)
    ax_iou.set_ylim(0, 1.05)

    ax_iou.plot(epochs, history["iou"], color=ACCENT_A, lw=LINE_W,
                marker="D", markersize=MARKER_S)
    ax_iou.fill_between(epochs, history["iou"], alpha=0.12, color=ACCENT_A)
    _best_epoch_marker(ax_iou, epochs, history["iou"], ACCENT_A, "iou")

    # ── 4. Dice panel ─────────────────────────────────────────────────────────
    _style_ax(ax_dice, "Dice Coefficient", "Dice", epochs)
    ax_dice.set_ylim(0, 1.05)

    ax_dice.plot(epochs, history["dice"], color=ACCENT_C, lw=LINE_W,
                 marker="o", markersize=MARKER_S)
    ax_dice.fill_between(epochs, history["dice"], alpha=0.12, color=ACCENT_C)
    _best_epoch_marker(ax_dice, epochs, history["dice"], ACCENT_C, "dice")

    # ── 5. F1 (derived) panel ─────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p  = np.array(history["precision"])
        r  = np.array(history["recall"])
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)

    _style_ax(ax_f1, "F1 Score  (derived)", "F1", epochs)
    ax_f1.set_ylim(0, 1.05)

    ax_f1.plot(epochs, f1, color="#79c0ff", lw=LINE_W,
               marker="s", markersize=MARKER_S)
    ax_f1.fill_between(epochs, f1, alpha=0.12, color="#79c0ff")
    _best_epoch_marker(ax_f1, epochs, f1.tolist(), "#79c0ff", "f1")

    # ── super-title ───────────────────────────────────────────────────────────
    sub = f" — {region}" if region else ""
    fig.text(
        0.5, 0.945,
        f"Trail Segmentation · Training Diagnostics{sub}",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=TEXT_MAIN,
        fontfamily="monospace",
    )
    fig.text(
        0.5, 0.918,
        f"{n_epochs} epochs · metrics recorded per epoch",
        ha="center", va="center",
        fontsize=8, color=TEXT_DIM, fontfamily="monospace",
    )

    # ── save ──────────────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics plot saved → {save_path}")

    return fig


# ── convenience wrapper that mirrors your project's config pattern ─────────────

def plot_metrics_from_config(history: dict, config: dict, save: bool = True) -> plt.Figure:
    """
    Thin wrapper to integrate with load_region_config() output.

    history : same dict of lists as plot_training_metrics().
    config  : dict returned by load_region_config().
    save    : if True, writes PNG next to other outputs in data/.
    """
    region   = config.get("active_region", "")
    tile     = config.get("active_tile_id", "")
    root     = config.get("project_root", os.getcwd())
    out_path = None

    if save:
        fname    = f"{region}_tile_{tile}_training_metrics.png" if (region or tile) else "training_metrics.png"
        out_path = os.path.join(root, "data", fname)

    fig = plot_training_metrics(history, save_path=out_path, region=region)
    plt.show()
    return fig


# ── quick smoke-test with synthetic data ──────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N   = 40

    def _sigmoid_trend(start, end, noise=0.02):
        x = np.linspace(-6, 6, N)
        trend = start + (end - start) * (1 / (1 + np.exp(-x)))
        return np.clip(trend + rng.normal(0, noise, N), 0, 1).tolist()

    synthetic = {
        "train_loss": (1 - np.array(_sigmoid_trend(0, 0.85))).tolist(),
        "val_loss":   (1 - np.array(_sigmoid_trend(0, 0.78))).tolist(),
        "iou":        _sigmoid_trend(0.10, 0.72),
        "dice":       _sigmoid_trend(0.18, 0.81),
        "precision":  _sigmoid_trend(0.30, 0.85),
        "recall":     _sigmoid_trend(0.25, 0.79),
    }

    fig = plot_training_metrics(
        synthetic,
        save_path="data/demo_training_metrics.png",
        region="Demo Region",
    )
    plt.show()