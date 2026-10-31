"""Génère les schémas d'architecture du projet et d'EEGNet EXP-04."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"

INK = "#0F172A"
MUTED = "#475569"
GRID = "#CBD5E1"
BG = "#F8FAFC"

PALETTE = {
    "data": ("#FEF3C7", "#D97706", "#78350F"),
    "prep": ("#E0F2FE", "#0284C7", "#0C4A6E"),
    "valid": ("#EDE9FE", "#7C3AED", "#4C1D95"),
    "lgb": ("#DCFCE7", "#16A34A", "#14532D"),
    "dl": ("#FFE4E6", "#E11D48", "#881337"),
    "post": ("#F1F5F9", "#475569", "#1E293B"),
    "app": ("#DBEAFE", "#2563EB", "#1E3A8A"),
    "head": ("#CCFBF1", "#0D9488", "#134E4A"),
}


def setup() -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(18, 11.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 100, 100, color=BG, zorder=0))
    return fig, ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    palette: tuple[str, str, str],
    *,
    title_size: float = 11.0,
    line_size: float = 8.9,
    spacing: float = 2.25,
) -> None:
    face, edge, title_color = palette
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.6, y - 0.6),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=1.6",
            facecolor=INK,
            edgecolor="none",
            alpha=0.08,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=1.6",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8,
            zorder=2,
        )
    )
    ax.text(
        x + 2.0,
        y + h - 2.55,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 2.0,
            y + h - 5.35 - i * spacing,
            line,
            ha="left",
            va="center",
            fontsize=line_size,
            color=INK,
            zorder=3,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    rad: float = 0.0,
    lw: float = 2.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def title_block(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(
        5,
        96.2,
        title,
        ha="left",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    ax.text(5, 92.2, subtitle, ha="left", va="center", fontsize=11.2, color=MUTED)
    ax.plot([5, 95], [89.6, 89.6], color=GRID, linewidth=1.0, zorder=1)


def global_schema() -> None:
    fig, ax = setup()
    title_block(
        ax,
        "Architecture globale du projet",
        "Pipeline P0 causal, validation strictement imbriquée et deux moteurs de détection",
    )

    box(
        ax,
        2,
        75,
        24,
        13.5,
        "1 · Données",
        [
            "22 enregistrements EEG",
            "19 patients annotés · 82 crises",
            "8 canaux EEG communs",
        ],
        PALETTE["data"],
    )
    box(
        ax,
        30,
        75,
        31,
        13.5,
        "2 · Prétraitement causal P0",
        [
            "Conversion .mat → .npy (v3_causal)",
            "Passe-bande 0,5–40 Hz sosfilt + notch",
            "Fenêtres 2 s · recouvrement 50 %",
            "33 206 fenêtres étiquetées",
        ],
        PALETTE["prep"],
        line_size=8.5,
    )
    box(
        ax,
        65,
        75,
        33,
        13.5,
        "3 · Validation strictement imbriquée",
        [
            "GroupKFold 5 plis par patient",
            "Validation interne : 25 % des patients",
            "Seuil τ* et early stopping sans test",
            "Test externe évalué une seule fois",
        ],
        PALETTE["valid"],
        line_size=8.5,
    )

    box(
        ax,
        8,
        51,
        40,
        16,
        "LightGBM EXP-00 · référence",
        [
            "260 features manuelles · 100 arbres",
            "F1 fenêtre 0,705 · détection 93,9 %",
            "4,55 FA/h causal · latence 4,46 s",
            "Modèle retenu pour la détection",
        ],
        PALETTE["lgb"],
    )
    box(
        ax,
        52,
        51,
        40,
        16,
        "EEGNet EXP-04 · embarqué",
        [
            "Signal brut 8 × 512 · 1 489 paramètres",
            "F1 fenêtre 0,643 · détection 93,9 %",
            "8,88 FA/h causal · latence 3,86 s",
            "Extension Deep Learning retenue",
        ],
        PALETTE["dl"],
    )

    box(
        ax,
        12,
        29,
        76,
        13.5,
        "4 · Décision et post-traitement causal",
        [
            "Seuil τ* par pli : LightGBM 0,63 ± 0,23  ·  EEGNet 0,77 ± 0,15",
            "Lissage médian causal → min gap → min durée → événements",
            "Aucune fusion rétroactive des trous",
        ],
        PALETTE["post"],
        line_size=9.1,
    )

    box(
        ax,
        10,
        9,
        38,
        13.5,
        "5 · Évaluation événementielle",
        [
            "Détection 93,9 % (77/82)",
            "F1 fenêtre 0,705 vs 0,643",
            "FA/h 4,55 vs 8,88 · latence 4,46 vs 3,86 s",
        ],
        PALETTE["valid"],
        line_size=8.6,
    )
    box(
        ax,
        52,
        9,
        38,
        13.5,
        "6 · Restitution",
        [
            "Application Streamlit multi-modèles",
            "Bascule LightGBM / EEGNet",
            "Rapports P0 · 130 tests",
        ],
        PALETTE["app"],
        line_size=8.8,
    )

    arrow(ax, (26, 81.7), (30, 81.7))
    arrow(ax, (61, 81.7), (65, 81.7))
    arrow(ax, (70, 75), (43, 67), rad=0.06)
    arrow(ax, (90, 75), (73, 67), rad=-0.04)
    arrow(ax, (28, 51), (38, 42.5), rad=0.03)
    arrow(ax, (72, 51), (62, 42.5), rad=-0.03)
    arrow(ax, (40, 29), (29, 22.5), rad=0.04)
    arrow(ax, (60, 29), (71, 22.5), rad=-0.04)

    ax.text(50, 4.2, "Protocole P0 : causal · imbriqué · test externe aveugle · 22 enregistrements / 9,23 h",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    fig.savefig(FIG_DIR / "07_architecture_globale.png", dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(FIG_DIR / "07_architecture_globale.svg", facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def dl_schema() -> None:
    fig, ax = setup()
    title_block(
        ax,
        "Architecture DL finale — EEGNet EXP-04",
        "Entrée brute 8 canaux × 512 échantillons · convolutions temporelle, spatiale et séparables",
    )

    main_x, main_w = 4, 60
    blocks = [
        (79, 8, "Entrée", "8 canaux EEG · 2 s · 256 Hz", "B × 1 × 8 × 512", PALETTE["data"]),
        (67, 8, "Bloc 1 · temporel", "Conv2D 1→8 · noyau 1×64 · padding 32 · BatchNorm", "B × 8 × 8 × 512", PALETTE["prep"]),
        (54, 9, "Bloc 2 · spatial depthwise", "Conv2D 8→16 · noyau 8×1 · groups=8 · BatchNorm · ELU · AvgPool 1×4 · Dropout 0,25", "B × 16 × 1 × 128", PALETTE["valid"]),
        (43, 7, "Bloc 3a · séparable depthwise", "Conv2D 16→16 · noyau 1×16 · padding 8 · groups=16", "B × 16 × 1 × 128", PALETTE["head"]),
        (31, 9, "Bloc 3b · pointwise", "Conv2D 16→16 · noyau 1×1 · BatchNorm · ELU · AvgPool 1×8 · Dropout 0,25", "B × 16 × 1 × 16", PALETTE["head"]),
        (22, 6, "Flatten", "Mise à plat des cartes spatio-temporelles", "B × 256", PALETTE["post"]),
        (14, 6, "Tête de classification", "Linear 256→1 · logit brut", "B × 1", PALETTE["lgb"]),
        (5, 7, "Score & décision", "Sigmoïde → seuil τ* par pli → événements", "B", PALETTE["dl"]),
    ]

    for y, h, title, detail, shape, palette in blocks:
        face, edge, title_color = palette
        ax.add_patch(
            FancyBboxPatch(
                (main_x + 0.55, y - 0.55),
                main_w,
                h,
                boxstyle="round,pad=0.02,rounding_size=1.3",
                facecolor=INK,
                edgecolor="none",
                alpha=0.08,
                zorder=1,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (main_x, y),
                main_w,
                h,
                boxstyle="round,pad=0.02,rounding_size=1.3",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.7,
                zorder=2,
            )
        )
        ax.text(main_x + 1.8, y + h - 2.1, title, ha="left", va="center",
                fontsize=9.8, fontweight="bold", color=title_color, zorder=3)
        ax.text(main_x + 1.8, y + 2.25, detail, ha="left", va="center",
                fontsize=7.55, color=INK, zorder=3)
        ax.text(main_x + main_w - 1.8, y + h - 2.1, shape, ha="right", va="center",
                fontsize=8.2, family="DejaVu Sans Mono", color=edge, zorder=3)

    for (y1, _, _, _, _, _), (y2, h2, _, _, _, _) in zip(blocks[:-1], blocks[1:]):
        arrow(ax, (main_x + main_w / 2, y1), (main_x + main_w / 2, y2 + h2), lw=1.9)

    panel_x, panel_w = 68, 29

    ax.add_patch(
        FancyBboxPatch(
            (panel_x, 45),
            panel_w,
            42,
            boxstyle="round,pad=0.02,rounding_size=1.5",
            facecolor="#FFFFFF",
            edgecolor=GRID,
            linewidth=1.4,
            zorder=2,
        )
    )
    ax.text(panel_x + 2, 83.5, "Entraînement P0", ha="left", va="center",
            fontsize=12, fontweight="bold", color=INK, zorder=3)
    training = [
        "FoldScaler ajusté sur le train uniquement",
        "Validation interne : 25 % des patients",
        "BCEWithLogitsLoss pondérée (pos_weight ≈ 27)",
        "Adam · lr 1e-3 · weight_decay 1e-4",
        "Batch 64 · 10 epochs maximum",
        "Early stopping sur validation interne",
        "Réentraînement sur le train externe complet",
        "Test externe évalué une seule fois",
    ]
    for i, line in enumerate(training):
        y = 79.2 - i * 4.25
        ax.plot([panel_x + 2.0], [y], marker="o", markersize=4.2, color="#7C3AED", zorder=3)
        ax.text(panel_x + 3.8, y, line, ha="left", va="center", fontsize=8.0, color=INK, zorder=3)

    ax.add_patch(
        FancyBboxPatch(
            (panel_x, 4),
            panel_w,
            37,
            boxstyle="round,pad=0.02,rounding_size=1.5",
            facecolor="#FFF1F2",
            edgecolor="#E11D48",
            linewidth=1.4,
            zorder=2,
        )
    )
    ax.text(panel_x + 2, 37.2, "Point de fonctionnement", ha="left", va="center",
            fontsize=12, fontweight="bold", color="#881337", zorder=3)
    results = [
        ("1 489 paramètres", "5,96 Ko en float32"),
        ("Seuil τ*", "0,77 ± 0,15"),
        ("F1 fenêtre", "0,643 ± 0,077"),
        ("Détection", "93,9 % (77/82)"),
        ("Fausses alarmes", "8,88 FA/h causal"),
        ("Latence causale", "3,86 s"),
    ]
    for i, (label, value) in enumerate(results):
        y = 31.5 - i * 4.55
        ax.text(panel_x + 2, y, label, ha="left", va="center", fontsize=8.0, color=MUTED, zorder=3)
        ax.text(panel_x + panel_w - 2, y, value, ha="right", va="center", fontsize=8.6,
                fontweight="bold", color="#881337", zorder=3)

    ax.text(50, 4.0, "Sortie : score de crise par fenêtre, puis post-traitement causal et évaluation événementielle",
            ha="center", va="center", fontsize=8.8, color=MUTED)

    fig.savefig(FIG_DIR / "07_architecture_dl_eegnet.png", dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(FIG_DIR / "07_architecture_dl_eegnet.svg", facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    global_schema()
    dl_schema()
    print(f"Figures écrites dans {FIG_DIR}")


if __name__ == "__main__":
    main()
