"""Shared matplotlib style: thin marks, recessive grid, fixed categorical order."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# categorical slots in fixed order (never cycled / never reassigned by rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"  # sequential = one hue, light -> dark

# fixed colour per model so every figure uses the same identity
MODEL_COLORS = {
    "Persistence": "#8a8984",
    "Seasonal mean": "#b8b7b0",
    "Ridge": SERIES[0],
    "Random Forest": SERIES[1],
    "Gradient Boosting": SERIES[2],
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    })


def save(fig, name: str, root) -> None:
    out = root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.show()
