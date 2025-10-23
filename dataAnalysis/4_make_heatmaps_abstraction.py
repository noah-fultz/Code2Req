

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


def plot_abstraction_detail_hybrid_heatmaps(summary: pd.DataFrame, out_path: Path):
    # === Consistent ordering ===
    project_order = ["eANCI", "SMOS", "eTour", "iTrust"]
    column_order = ["Feature", "Feature + RAG", "Code", "Code + RAG"]

    # === Label map ===
    label_map = {
        "qwen-feature": "Feature",
        "qwen-feature-rag": "Feature + RAG",
        "qwen-code": "Code",
        "qwen-code-rag": "Code + RAG",
        "gpt-feature": "Feature",
        "gpt-feature-rag": "Feature + RAG",
        "gpt-code": "Code",
        "gpt-code-rag": "Code + RAG",
    }

    # === Split by model ===
    df_qwen = summary[summary["variation"].str.startswith("qwen")].copy()
    df_gpt = summary[summary["variation"].str.startswith("gpt")].copy()

    # === Compute percentages ===
    def pct(df, col):
        return (df[col] / df["total_reqs"].replace({0: np.nan})) * 100

    def pct_of_subset(df, subset_col, base_col):
        """% of subset_col relative to base_col (e.g., matched of generated)"""
        return (df[subset_col] / df[base_col].replace({0: np.nan})) * 100

    for base in [
        "num_mapped_abstraction_lower",
        "num_mapped_abstraction_equal",
        "num_mapped_abstraction_higher",
        "num_mapped_detail_lower",
    ]:
        overall = base.replace("num_mapped_", "num_")
        for df in (df_qwen, df_gpt):
            df[f"{overall}_pct"] = pct(df, overall)  # % of all generated
            df[f"{base}_within_pct"] = pct_of_subset(df, base, overall)  # % of those that were matched

    # === Pivot helper ===
    def make_pivot(df, base_col, within_col):
        base_pivot = (
            df.pivot(index="project", columns="variation", values=base_col)
            .rename(columns=label_map)
            .reindex(columns=column_order)
            .reindex(project_order)
        )
        within_pivot = (
            df.pivot(index="project", columns="variation", values=within_col)
            .rename(columns=label_map)
            .reindex(columns=column_order)
            .reindex(project_order)
        )
        base_pivot.index.name = within_pivot.index.name = None
        return base_pivot, within_pivot

    # === Prepare pivots ===
    pivots_qwen, pivots_gpt = {}, {}
    combos = [
        ("Abstraction lower", "num_abstraction_lower_pct", "num_mapped_abstraction_lower_within_pct"),
        ("Abstraction equal", "num_abstraction_equal_pct", "num_mapped_abstraction_equal_within_pct"),
        ("Abstraction higher", "num_abstraction_higher_pct", "num_mapped_abstraction_higher_within_pct"),
        ("Detail lower", "num_detail_lower_pct", "num_mapped_detail_lower_within_pct"),
    ]

    for title, base_col, within_col in combos:
        pivots_qwen[title] = make_pivot(df_qwen, base_col, within_col)
        pivots_gpt[title] = make_pivot(df_gpt, base_col, within_col)

    # === Plot setup ===
    sns.set_theme(style="white")
    cmap = sns.color_palette("viridis", as_cmap=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(4, 2, figsize=(8, 9.5), sharey=True)
    vmin, vmax = 0, 100

    # === Pretty hybrid cell renderer ===
    def draw_heatmap_with_bars(ax, base_df, within_df):
        sns.heatmap(base_df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, linewidths=0.6, linecolor="white")

        for i in range(base_df.shape[0]):
            for j in range(base_df.shape[1]):
                val_base = base_df.iloc[i, j]
                val_match = within_df.iloc[i, j]

                # Ensure numeric and display zeros explicitly
                if pd.isna(val_base): val_base = 0
                if pd.isna(val_match): val_match = 0

                # Clip to [0,100]
                val_base = np.clip(val_base, 0, 100)
                val_match = np.clip(val_match, 0, 100)
                frac = val_match / 100

                # Add rounded black bar (prettier overlay)
                bar = FancyBboxPatch(
                    (j, i + 0.83), frac, 0.12,
                    boxstyle="round,pad=0.02,rounding_size=0.02",
                    linewidth=0, facecolor="black", alpha=0.55,
                    transform=ax.transData
                )
                ax.add_patch(bar)

                # Annotation text
                color = "white" if val_base < 40 else "black"
                ax.text(
                    j + 0.5, i + 0.38,
                    f"{val_base:.0f} ({val_match:.0f}%)",
                    ha="center", va="center",
                    color=color, fontsize=8
                )

    # === Titles ===
    titles_qwen = [
        "A. Abstraction lower (Qwen2.5-coder)",
        "C. Abstraction equal (Qwen2.5-coder)",
        "E. Abstraction higher (Qwen2.5-coder)",
        "G. Detail lower (Qwen2.5-coder)",
    ]
    titles_gpt = [
        "B. Abstraction lower (GPT-5)",
        "D. Abstraction equal (GPT-5)",
        "F. Abstraction higher (GPT-5)",
        "H. Detail lower (GPT-5)",
    ]

    # === Draw heatmaps ===
    for row, (metric, _, _) in enumerate(combos):
        (base_q, within_q) = pivots_qwen[metric]
        (base_g, within_g) = pivots_gpt[metric]

        # Left (Qwen)
        ax = axes[row, 0]
        draw_heatmap_with_bars(ax, base_q, within_q)
        ax.set_title(titles_qwen[row], pad=7)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

        # Right (GPT)
        ax = axes[row, 1]
        draw_heatmap_with_bars(ax, base_g, within_g)
        ax.set_title(titles_gpt[row], pad=7)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

    # === Shared colorbar ===
    cbar_ax = fig.add_axes([0.91, 0.38, 0.015, 0.22])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("% of all generated requirements", rotation=90, labelpad=10)

    # === Layout ===
    plt.subplots_adjust(hspace=0.65, wspace=0.15, right=0.9, top=0.98, bottom=0.06)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved polished hybrid abstraction/detail heatmap to {out_path}")


# === Main ===
if __name__ == "__main__":
    summary_path = Path("manual_analysis/summary_stats.csv")
    out_path = Path("manual_analysis/plots/heatmap_abstraction_detail.png")

    summary = pd.read_csv(summary_path)
    plot_abstraction_detail_hybrid_heatmaps(summary, out_path)
