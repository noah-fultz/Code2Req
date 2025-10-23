
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# --- Helper functions
# ============================================================

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    header = df.iloc[1].astype(str).str.strip()
    df = df.iloc[2:].copy()
    df.columns = header
    df = df.loc[:, ~(df.isna().all(axis=0))]
    df.columns = [c.strip() for c in df.columns]
    return df.reset_index(drop=True)


def load_project_variation(excel_path: str, sheet_name: str) -> pd.DataFrame:
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    return _clean_headers(df_raw)


def compute_metrics(df: pd.DataFrame, total_reqs_gs: int) -> dict:
    """Compute manual F1 and percentage of fully matched secondary concepts (over all generated requirements)."""
    cols = {c.lower(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    col_manmatch = get_col("Key concept manually matched to")
    col_secondary = get_col("Secondary concept match")

    total = len(df)
    if total == 0:
        return {"f1": np.nan, "secondary_yes_pct": np.nan}

    # --- Manual matches ---
    s_match = df[col_manmatch].fillna("").astype(str).str.strip()
    man_match_yes = int((s_match != "").sum())
    man_match_no = int((s_match == "").sum())

    all_reqs = {
        tok.strip()
        for cell in s_match if cell.strip()
        for tok in cell.split(",") if tok.strip()
    }

    TP = man_match_yes
    FP = man_match_no
    FN = total_reqs_gs - len(all_reqs)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- Secondary concept matches ---
        # --- Secondary concept matches ---
    sec_yes = 0
    sec_partly = 0
    sec_no = 0

    if col_secondary in df.columns and col_manmatch in df.columns:
        s = df[col_secondary].fillna("").astype(str).str.strip().str.lower()
        matched_mask = s_match != ""  # only count for manually matched reqs
        for val, matched in zip(s, matched_mask):
            if not matched:
                continue
            tokens = [t.strip().lower() for t in val.split(",") if t.strip()]
            # Prioritize yes > partly > no
            if any("yes" in t for t in tokens):
                sec_yes += 1
            elif any("partly" in t for t in tokens):
                sec_partly += 1
            elif any("no" in t for t in tokens):
                sec_no += 1

    # IMPORTANT: divide by ALL generated requirements, but count only matched
    sec_yes_pct = (sec_yes / total) * 100 if total > 0 else np.nan


    return {"f1": f1, "secondary_yes_pct": sec_yes_pct, "total_reqs": total}


# ============================================================
# --- Build summary across projects and variants
# ============================================================

def build_summary(excel_path: str, total_reqs_gs: dict) -> pd.DataFrame:
    projects = ["eANCI", "SMOS", "eTour", "iTrust"]
    variations = {
        "qwen-feature": "Feature_Qwen",
        "qwen-feature-rag": "Feature_RAG_Qwen",
        "qwen-code": "Code_Qwen",
        "qwen-code-rag": "Code_RAG_Qwen",
        "gpt-feature": "Feature_GPT",
        "gpt-feature-rag": "Feature_RAG_GPT",
        "gpt-code": "Code_GPT",
        "gpt-code-rag": "Code_RAG_GPT",
    }

    records = []
    for proj in projects:
        for var_label, suffix in variations.items():
            sheet = f"{proj}_{suffix}"
            try:
                df = load_project_variation(excel_path, sheet)
            except Exception:
                print(f"⚠️ Skipping missing sheet: {sheet}")
                continue
            m = compute_metrics(df, total_reqs_gs[proj])
            m.update({"project": proj, "variation": var_label})
            records.append(m)

    summary = pd.DataFrame(records)
    proj_order = ["eANCI", "SMOS", "eTour", "iTrust"]
    var_order = list(variations.keys())
    summary["project"] = pd.Categorical(summary["project"], categories=proj_order, ordered=True)
    summary["variation"] = pd.Categorical(summary["variation"], categories=var_order, ordered=True)
    return summary.sort_values(["project", "variation"]).reset_index(drop=True)


# ============================================================
# --- Plot 2x2 grid of heatmaps
# ============================================================
def plot_split_heatmaps(summary: pd.DataFrame, out_path: Path):
    df = summary.copy()
    df_qwen = df[df["variation"].str.startswith("qwen")]
    df_gpt = df[df["variation"].str.startswith("gpt")]

    # Pivot to wide format
    f1_qwen = df_qwen.pivot(index="project", columns="variation", values="f1")
    f1_gpt = df_gpt.pivot(index="project", columns="variation", values="f1")
    sec_qwen = df_qwen.pivot(index="project", columns="variation", values="secondary_yes_pct")
    sec_gpt = df_gpt.pivot(index="project", columns="variation", values="secondary_yes_pct")

    for table in (f1_qwen, f1_gpt, sec_qwen, sec_gpt):
            table.index.name = None
    # --- Simplified x-axis labels ---
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
    f1_qwen = f1_qwen.rename(columns=label_map)
    f1_gpt = f1_gpt.rename(columns=label_map)
    sec_qwen = sec_qwen.rename(columns=label_map)
    sec_gpt = sec_gpt.rename(columns=label_map)

    # --- Font scaling for compact but readable layout ---
    plt.rcParams.update({
        "font.size": 11,          # base font
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
        "legend.fontsize": 10
    })

    # Smaller figure — keeps everything denser and fonts proportionally larger
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    cmap = sns.color_palette("viridis", as_cmap=True)

    # --- F1 heatmaps ---
    sns.heatmap(f1_qwen, ax=axes[0, 0], cmap=cmap, vmin=0, vmax=1,
                annot=True, fmt=".2f", cbar=False, linewidths=0.5,
                annot_kws={"size": 10, "color": "white"})
    sns.heatmap(f1_gpt, ax=axes[0, 1], cmap=cmap, vmin=0, vmax=1,
                annot=True, fmt=".2f", cbar=True,
                cbar_kws={"label": "F1-score"}, linewidths=0.5,
                annot_kws={"size": 10, "color": "white"})

    axes[0, 0].set_title("A. F1-score - Qwen2.5-coder", pad=6)
    axes[0, 1].set_title("B. F1-score - GPT-5", pad=6)

    # --- Secondary match heatmaps ---
    sns.heatmap(sec_qwen, ax=axes[1, 0], cmap=cmap, vmin=0, vmax=100,
                annot=True, fmt=".0f", cbar=False, linewidths=0.5,
                annot_kws={"size": 10, "color": "white"})
    sns.heatmap(sec_gpt, ax=axes[1, 1], cmap=cmap, vmin=0, vmax=100,
                annot=True, fmt=".0f", cbar=True,
                cbar_kws={"label": "% fully matched (of all generated)"}, linewidths=0.5,
                annot_kws={"size": 10, "color": "white"})

    axes[1, 0].set_title("C. Fully matched secondary concepts (%) - Qwen2.5-coder", pad=6)
    axes[1, 1].set_title("D. Fully matched secondary concepts (%) - GPT-5", pad=6)

    # --- Axis labels ---
    for ax in axes.flat:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    axes[0, 0].set_ylabel("")
    axes[1, 0].set_ylabel("")

    # fig.suptitle("Requirement generation performance across projects", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved compact, high-legibility heatmap grid to {out_path}")

# ============================================================
# --- Main
# ============================================================

if __name__ == "__main__":
    excel_path = "Manual_analysis.xlsx"
    out_dir = Path("manual_analysis/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_reqs_gs = {"eANCI": 39, "SMOS": 67, "eTour": 57, "iTrust": 105}

    summary = build_summary(excel_path, total_reqs_gs)
    plot_split_heatmaps(summary, out_dir / "heatmaps_f1_secondary.png")
