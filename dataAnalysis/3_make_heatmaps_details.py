
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_adherence_heatmaps_vertical(summary: pd.DataFrame, out_path: Path):
    metric_map = {
        "Syntactic adherence (%)": "num_follows_format_yes",
        "Semantic adherence (%)": "num_pure_system_yes",
        "Existing in code (%)": "num_exists_in_code_yes",
        "Recommended addition (%)": "num_recommended_additions",
    }

    # --- Compute percentages ---
    df = summary.copy()
    for col in metric_map.values():
        if col in df.columns:
            df[f"{col}_pct"] = (df[col] / df["total_reqs"].replace({0: np.nan})) * 100

    # --- Split by model ---
    df_qwen = df[df["variation"].str.startswith("qwen")].copy()
    df_gpt = df[df["variation"].str.startswith("gpt")].copy()

    # --- Clean labels ---
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
    project_order = ["eANCI", "SMOS", "eTour", "iTrust"]
    column_order = ["Feature", "Feature + RAG", "Code", "Code + RAG"]

    pivot_qwen, pivot_gpt = {}, {}
    for metric_title, col in metric_map.items():
        pq = df_qwen.pivot(index="project", columns="variation", values=f"{col}_pct")
        pg = df_gpt.pivot(index="project", columns="variation", values=f"{col}_pct")

        # Enforce consistent project and column order
        pq = pq.rename(columns=label_map).reindex(columns=column_order)
        pg = pg.rename(columns=label_map).reindex(columns=column_order)
        pq = pq.reindex(project_order)
        pg = pg.reindex(project_order)
        pq.index.name = None
        pg.index.name = None

        pivot_qwen[metric_title] = pq
        pivot_gpt[metric_title] = pg

    # --- Plot setup ---
    sns.set_theme(style="white")
    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(4, 2, figsize=(7.5, 9.0), sharey=True)
    vmin, vmax = 0, 100

    # Adaptive text color for legibility
    def annotate_heatmap(data, ax, fmt=".0f"):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.iloc[i, j]
                if pd.isna(val):
                    continue
                color = "black" if val > 75 else "white"
                ax.text(j + 0.5, i + 0.5, format(val, fmt),
                        ha="center", va="center", color=color, fontsize=10)

    # Subplot titles
    titles_qwen = [
        "A. Syntactic adherence (Qwen2.5-coder)",
        "C. Semantic adherence (Qwen2.5-coder)",
        "E. Existing in code (Qwen2.5-coder)",
        "G. Recommended addition (Qwen2.5-coder)",
    ]
    titles_gpt = [
        "B. Syntactic adherence (GPT-5)",
        "D. Semantic adherence (GPT-5)",
        "F. Existing in code (GPT-5)",
        "H. Recommended addition (GPT-5)",
    ]

    # --- Draw heatmaps ---
    for row_idx, metric_title in enumerate(metric_map.keys()):
        pq = pivot_qwen[metric_title]
        pg = pivot_gpt[metric_title]

        # Left: Qwen
        ax = axes[row_idx, 0]
        sns.heatmap(pq, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, linewidths=0.6, linecolor="white",
                    square=False)
        annotate_heatmap(pq, ax)
        ax.set_title(titles_qwen[row_idx], pad=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Rotate tick labels for x and y
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

        # Right: GPT
        ax = axes[row_idx, 1]
        sns.heatmap(pg, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, linewidths=0.6, linecolor="white",
                    square=False)
        annotate_heatmap(pg, ax)
        ax.set_title(titles_gpt[row_idx], pad=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

    # --- Shared, compact colorbar ---
    cbar_ax = fig.add_axes([0.91, 0.38, 0.015, 0.22])  # smaller + centered
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("% of all generated requirements", rotation=270, labelpad=15)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.6, wspace=0.15, right=0.88, top=0.98, bottom=0.06)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved final adherence heatmap grid with vertical project labels to {out_path}")


# ============================================================
# --- Main
# ============================================================

if __name__ == "__main__":
    summary_path = Path("manual_analysis/summary_stats.csv")
    out_path = Path("manual_analysis/plots/heatmap_adherence.png")

    summary = pd.read_csv(summary_path)
    plot_adherence_heatmaps_vertical(summary, out_path)
