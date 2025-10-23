import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colors as mcolors

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Use the first row as the header, strip whitespace, and drop empty columns."""
    header = df.iloc[1].astype(str).str.strip()
    df = df.iloc[2:].copy()
    df.columns = header
    # Drop columns that are entirely NaN after header shift
    df = df.loc[:, ~(df.isna().all(axis=0))]
    # Normalize column names a bit
    df.columns = [c.strip() for c in df.columns]
    return df.reset_index(drop=True)

def _yes_no_to_bool(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})

def _norm_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def load_project_variation(excel_path: str, sheet_name: str) -> pd.DataFrame:
    print("getting file and sheet "+str(excel_path)+" : "+str(sheet_name))
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    df = _clean_headers(df_raw)
    return df

def compute_metrics(df: pd.DataFrame, total_reqs_gs) -> Dict[str, int]:
    # Standardize expected columns (be resilient to slight name variants)
    cols = {c.lower(): c for c in df.columns}
    def get_col(*cands):
        for cand in cands:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    col_pure = get_col("Semantic adherence")
    col_format = get_col("Syntactic adherence")
    col_abst = get_col("Abstraction level")
    col_detail = get_col("Detail Level")
    col_reco = get_col("Recommended addition")
    col_manmatch = get_col("Key concept manually matched to")
    col_secondary = get_col("Secondary concept match")
    col_code = get_col("Key concept code class")

    total = len(df)

    # Booleans
    def col_bool(colname):
        if colname is None or colname not in df:
            return pd.Series([False]*len(df))
        return df[colname].pipe(_yes_no_to_bool).fillna(False)

    # Text levels
    def col_text(colname):
        if colname is None or colname not in df:
            return pd.Series([""]*len(df))
        return _norm_text(df[colname])

    pure = col_bool(col_pure)
    follows = col_bool(col_format)

    abst = col_text(col_abst)  # expect 'higher', 'equal', 'lower'
    detail = col_text(col_detail)

    code_count_yes = 0
    code_count_no = 0
    if col_code and col_code in df:
        # Work on the raw series to preserve NA (avoid astype(str) -> "nan")
        s = df[col_code].astype("string").str.strip()

        # Placeholders that should count as "no" even if non-empty
        placeholders = {"", "na", "n/a", "none", "-"}

        # Build masks
        placeholder_mask = s.str.lower().isin(placeholders)
        mask_yes = s.notna() & ~placeholder_mask           # real text present
        mask_no  = s.isna() | placeholder_mask             # empty/placeholder/NA

        code_count_yes = int(mask_yes.sum())
        code_count_no  = int(mask_no.sum())

    # Recommended addition: count non-empty and not 'na'
    if col_reco and col_reco in df:
        reco_norm = col_text(col_reco)
        reco_count_yes=(reco_norm.notna() & (reco_norm.isin(["Yes", "yes"]))).sum()

    man_match_yes = 0
    man_match_no  = 0
    all_reqs = set()

    if col_manmatch and col_manmatch in df:
        # Keep it simple: empty vs non-empty after stripping
        s = df[col_manmatch].fillna("").astype(str)

        man_match_yes = int((s.str.strip() != "").sum())
        man_match_no  = int((s.str.strip() == "").sum())

        # Build the set of all IDs (comma-separated in each cell)
        all_reqs = {
            tok.strip()
            for cell in s
            if cell.strip()
            for tok in cell.split(",")
            if tok.strip()
        }
    else:
        man_match_yes = 0
        man_match_no  = 0
        all_reqs = set()

    TP = man_match_yes  # true positives: correct matches
    FP = man_match_no  # false positives: generated items that were wrong
    FN = total_reqs_gs-len(all_reqs)  # false negatives: gold items you missed
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0)

    num_mapped_abst_equal = 0
    num_mapped_abst_lower=0
    num_mapped_abst_higher=0

    num_mapped_detail_equal = 0
    num_mapped_detail_lower=0
    num_mapped_detail_higher=0

    if col_manmatch and col_manmatch in df and col_abst and col_abst in df:
        # mapped = non-empty manual mapping
        mapped_mask = df[col_manmatch].fillna("").astype(str).str.strip() != ""

        # abstraction == "Equal" (case/whitespace insensitive)
        abst_equal_mask = df[col_abst].astype(str).str.strip().str.lower() == "equal"
        abst_lower_mask = df[col_abst].astype(str).str.strip().str.lower() == "lower"
        abst_higher_mask = df[col_abst].astype(str).str.strip().str.lower() == "higher"

        detail_equal_mask = df[col_detail].astype(str).str.strip().str.lower() == "equal"
        detail_lower_mask = df[col_detail].astype(str).str.strip().str.lower() == "lower"
        detail_higher_mask = df[col_detail].astype(str).str.strip().str.lower() == "higher"

        num_mapped_abst_equal = int((mapped_mask & abst_equal_mask).sum())
        num_mapped_abst_lower = int((mapped_mask & abst_lower_mask).sum())
        num_mapped_abst_higher = int((mapped_mask & abst_higher_mask).sum())

        num_mapped_detail_equal = int((mapped_mask & detail_equal_mask).sum())
        num_mapped_detail_lower = int((mapped_mask & detail_lower_mask).sum())
        num_mapped_detail_higher = int((mapped_mask & detail_higher_mask).sum())

    # --- unmapped (no manual match) with Abstraction == "Equal" ---
    num_unmapped_abst_equal = 0
    num_unmapped_abst_lower=0
    num_unmapped_abst_higher=0

    num_unmapped_detail_equal = 0
    num_unmapped_detail_lower=0
    num_unmapped_detail_higher=0

    if col_manmatch and col_manmatch in df:
        mapped_mask = df[col_manmatch].fillna("").astype(str).str.strip() != ""
    else:
        mapped_mask = pd.Series(False, index=df.index)

    unmapped_mask = ~mapped_mask
    num_unmapped_abst_equal = int((unmapped_mask & abst_equal_mask).sum())
    num_unmapped_abst_lower = int((unmapped_mask & abst_lower_mask).sum())
    num_unmapped_abst_higher = int((unmapped_mask & abst_higher_mask).sum())

    num_unmapped_detail_equal = int((unmapped_mask & detail_equal_mask).sum())
    num_unmapped_detail_lower = int((unmapped_mask & detail_lower_mask).sum())
    num_unmapped_detail_higher = int((unmapped_mask & detail_higher_mask).sum())
        # --- Secondary concept match (Yes / Partly / No), only for manually matched ---
    sec_yes = sec_partly = sec_no = 0
    if col_secondary and col_secondary in df and col_manmatch and col_manmatch in df:
        s = df[col_secondary].fillna("").astype(str).str.strip().str.lower()
        matched_mask = df[col_manmatch].fillna("").astype(str).str.strip() != ""

        for val, matched in zip(s, matched_mask):
            if not matched:
                continue
            tokens = [t.strip() for t in val.split(",") if t.strip()]
            tokens = [t.lower() for t in tokens]

            if "yes" in tokens:
                sec_yes += 1
            elif "partly" in tokens:
                sec_partly += 1
            elif "no" in tokens:
                sec_no += 1

    metrics = {
        "total_reqs": int(total),
        "num_exists_in_code_yes": int(code_count_yes),
        "num_exists_in_code_no": int(code_count_no),
        "num_pure_system_yes": int((pure == True).sum()),
        "num_pure_system_no": int((pure == False).sum()),
        "num_fully_matched_yes": int(man_match_yes),
        "num_not_matched_yes": int(man_match_no),
        "num_follows_format_no": int((follows == False).sum()),
        "num_follows_format_yes": int((follows == True).sum()),
        "num_abstraction_higher": int((abst == "higher").sum()),
        "num_abstraction_equal": int((abst == "equal").sum()),
        "num_abstraction_lower": int((abst == "lower").sum()),
        "num_detail_higher": int((detail == "higher").sum()),
        "num_detail_equal": int((detail == "equal").sum()),
        "num_detail_lower": int((detail == "lower").sum()),
        "num_recommended_additions": int(reco_count_yes),
        "num_mapped_abstraction_equal": int(num_mapped_abst_equal),
        "num_unmapped_abstraction_equal": int(num_unmapped_abst_equal),
        "num_mapped_abstraction_higher": int(num_mapped_abst_higher),
        "num_unmapped_abstraction_higher": int(num_unmapped_abst_higher),
        "num_mapped_abstraction_lower": int(num_mapped_abst_lower),
        "num_unmapped_abstraction_lower": int(num_unmapped_abst_lower),
        "num_mapped_detail_equal": int(num_mapped_detail_equal),
        "num_unmapped_detail_equal": int(num_unmapped_detail_equal),
        "num_mapped_detail_higher": int(num_mapped_detail_higher),
        "num_unmapped_detail_higher": int(num_unmapped_detail_higher),
        "num_mapped_detail_lower": int(num_mapped_detail_lower),
        "num_unmapped_detail_lower": int(num_unmapped_detail_lower),
        "f1": f1,
        "secondary_match_yes": sec_yes,
        "secondary_match_partly": sec_partly,
        "secondary_match_no": sec_no,
    }
    return metrics

def build_summary(excel_path: str, total_reqs_gs) -> pd.DataFrame:
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
            except Exception as e:
                print("Failed loading sheet!")
                print(str(e))
                continue
            m = compute_metrics(df, total_reqs_gs[proj])
            m.update({"project": proj, "variation": var_label})
            records.append(m)

    summary = pd.DataFrame.from_records(records)
    proj_order = ["eANCI", "SMOS", "eTour", "iTrust"]
    var_order = list(variations.keys())
    summary["project"] = pd.Categorical(summary["project"], categories=proj_order, ordered=True)
    summary["variation"] = pd.Categorical(summary["variation"], categories=var_order, ordered=True)
    summary = summary.sort_values(["project", "variation"]).reset_index(drop=True)
    return summary


def lighten(hex_color, factor=0.45):
    # blend color with white (factor in 0..1, higher = lighter)
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex((1 - factor) * rgb + factor * 1)

# base hues (match your figure: blue=feature, green=featureWithRAG, orange=code, red=codeWithRAG)
base = {
    "feature": "#1f77b4",        # blue
    "feature-rag": "#2ca02c",    # green
    "code": "#ff7f0e",           # orange
    "code-rag": "#d62728",       # red
}

total_reqs_gs={"eANCI":39, "SMOS":67,"eTour":57,"iTrust":105}
out_dir = Path("manual_analysis")
out_dir.mkdir(parents=True, exist_ok=True)
plots_dir = out_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

summary = build_summary("Manual_analysis.xlsx", total_reqs_gs)
summary_path = out_dir / "summary_stats.csv"
summary.to_csv(summary_path, index=False)

plot_files=[]

# --- NEW: Average manual F1 across all four projects (reuse existing colors) ---
target_projects = ["eANCI", "SMOS", "eTour", "iTrust"]
# target_projects = ["eANCI", "SMOS", "eTour"]
df_f1 = summary[summary["project"].isin(target_projects)].copy()

# Respect the categorical order set earlier; otherwise fall back to sorted uniques
if hasattr(summary["variation"], "cat"):
    var_order = summary["variation"].cat.categories.tolist()
else:
    var_order = sorted(df_f1["variation"].unique())

avg_f1 = (
    df_f1.groupby("variation", observed=True)["f1"]
    .mean()
    .reindex(var_order)
)
std_f1 = (
    df_f1.groupby("variation", observed=True)["f1"]
    .std()
    .reindex(var_order)
)

# Keep only variations that actually have data
mask = ~avg_f1.isna()
avg_f1 = avg_f1[mask]
std_f1 = std_f1[mask]
variations_present = avg_f1.index.tolist()

color_map = {}
for v in variations_present:
    parts = v.split("-")
    model = parts[0].lower()
    kind = "-".join(parts[1:]) or v
    color_map[v] = lighten(base[kind]) if model.startswith("qwen") else base[kind]

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(variations_present))
bars = ax.bar(
    x,
    avg_f1.values,
    color=[color_map[v] for v in variations_present],
    edgecolor="white",
    linewidth=0.5,
)
yerr = np.nan_to_num(std_f1.values, nan=0.0)
ax.errorbar(x, avg_f1.values, yerr=yerr, fmt="none", capsize=4, linewidth=1)

# Add numbers above bars
for bar, value, err in zip(bars, avg_f1.values, yerr):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + err + 0.02,  # bar height + error bar + offset
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Define a mapping from variation string → pretty label
label_map = {
    "qwen-feature": "From Feature - Qwen",
    "qwen-feature-rag": "From Feature with RAG - Qwen",
    "qwen-code": "From Code - Qwen",
    "qwen-code-rag": "From Code with RAG - Qwen",
    "gpt-feature": "From Feature - GPT",
    "gpt-feature-rag": "From Feature with RAG - GPT",
    "gpt-code": "From Code - GPT",
    "gpt-code-rag": "From Code with RAG - GPT",
}

# When setting xticklabels, replace with mapped names
ax.set_xticks(x)
ax.set_xticklabels([label_map.get(v, v) for v in variations_present], 
                   rotation=30, ha="right")

ax.set_ylabel("Average F1 score")
ax.set_title("Average F1 score across projects (eANCI, SMOS, eTour, iTrust)")
ax.set_ylim(0, 1.05)

plt.tight_layout()
avg_path = plots_dir / "avg_f1_over_projects.png"
plt.savefig(avg_path)
plt.close()
plot_files.append(avg_path)

# --- Average % lift in MANUAL F1 (with consistent colors) ---
df_f1_manual = summary[summary["project"].isin(target_projects)].copy()

# Compute percent lift: (RAG - baseline) / baseline * 100
def compute_lift(df, base, rag, metric="f1"):
    base_vals = df[df["variation"] == base].set_index("project")[metric]
    rag_vals = df[df["variation"] == rag].set_index("project")[metric]
    common_projects = base_vals.index.intersection(rag_vals.index)
    lifts = ((rag_vals.loc[common_projects] - base_vals.loc[common_projects]) /
             base_vals.loc[common_projects]) * 100
    return lifts

lifts = {
    "qwen-feature-rag": compute_lift(df_f1_manual, "qwen-feature", "qwen-feature-rag"),
    "qwen-code-rag": compute_lift(df_f1_manual, "qwen-code", "qwen-code-rag"),
    "gpt-feature-rag": compute_lift(df_f1_manual, "gpt-feature", "gpt-feature-rag"),
    "gpt-code-rag": compute_lift(df_f1_manual, "gpt-code", "gpt-code-rag"),
}

avg_lifts = {k: v.mean() for k, v in lifts.items()}
std_lifts = {k: v.std() for k, v in lifts.items()}

# Reuse consistent colors
color_map = {}
for v in avg_lifts.keys():
    parts = v.split("-")
    model = parts[0].lower()
    kind = "-".join(parts[1:]) or v
    color_map[v] = lighten(base[kind]) if model.startswith("qwen") else base[kind]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(avg_lifts))
bars = ax.bar(
    x,
    list(avg_lifts.values()),
    color=[color_map[v] for v in avg_lifts.keys()],
    edgecolor="white",
    linewidth=0.7
)

# Error bars
yerr = list(std_lifts.values())
ax.errorbar(
    x,
    list(avg_lifts.values()),
    yerr=yerr,
    fmt="none",
    capsize=5,
    color="black"
)

# Add labels ABOVE the error bars
for bar, val, err in zip(bars, avg_lifts.values(), yerr):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + err + 1,  # place above error bar
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Baseline at y=0
ax.axhline(0, color="black", linewidth=1, linestyle="--")

# Pretty labels
label_map = {
    "qwen-feature-rag": "From Feature - Qwen",
    "qwen-code-rag": "From Code - Qwen",
    "gpt-feature-rag": "From Feature - GPT",
    "gpt-code-rag": "From Code - GPT",
}
ax.set_xticks(x)
ax.set_xticklabels([label_map.get(v, v) for v in avg_lifts.keys()],
                   rotation=20, ha="right")

ax.set_ylabel("Avg. percent change in F1 score")
ax.set_title("Average percent change in F1 score using RAG across projects")

plt.tight_layout()
lift_path = plots_dir / "avg_percent_lift_manual_f1.png"
plt.savefig(lift_path)
plt.close()
plot_files.append(lift_path)
