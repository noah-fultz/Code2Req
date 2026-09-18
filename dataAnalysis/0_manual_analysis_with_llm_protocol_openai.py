"""
Variant of 0_manual_analysis_with_llm_protocol.py that talks to the public
OpenAI API (via an API key) instead of an Azure OpenAI deployment (via RBAC/
`az login`). Everything else - the verbatim-protocol system prompt, loading,
batching, checkpointing, agreement computation, output workbook layout - is
unchanged.
Uses separate output/checkpoint locations so this script doesn't clobber the
Azure-backed script's results:
  - Output workbook: Manual_analysis_llm_protocol_openai.xlsx
  - Checkpoints:      llm_checkpoints_protocol_openai/{dataset}_{variant}.csv

Auth: set the OPENAI_API_KEY environment variable (or edit OPENAI_API_KEY
below) before running. Get a key from https://platform.openai.com/api-keys.

See the module docstring in 0_manual_analysis_with_llm.py for input format
and usage details (identical here, aside from the client/auth setup below).

python 0_manual_analysis_with_llm_protocol_openai.py --agreement
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from openai import OpenAI
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows
from pydantic import BaseModel, Field, create_model
from sklearn.metrics import cohen_kappa_score

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = SCRIPT_DIR / "Manual_analysis.xlsx"
OUTPUT_XLSX = SCRIPT_DIR / "Manual_analysis_llm_protocol_openai.xlsx"
CHECKPOINT_DIR = SCRIPT_DIR / "llm_checkpoints_protocol_openai"

DATASETS = ["eANCI", "SMOS", "eTour", "iTrust"]
VARIANTS = [
    "Code_RAG_GPT",
    "Code_GPT",
    "Feature_RAG_GPT",
    "Feature_GPT",
    "Code_RAG_Qwen",
    "Code_Qwen",
    "Feature_RAG_Qwen",
    "Feature_Qwen",
]

# Reads the key from the OPENAI_API_KEY env var by default; fall back to a
# literal string here only if you don't want to set the env var.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.6-sol"

RETRIES = 3
RETRY_BACKOFF_SECONDS = 15


# ============================================================
# OpenAI client (API-key auth)
# ============================================================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
            "(or edit OPENAI_API_KEY in this script)."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# Response schema (mirrors the columns filled in during manual analysis).
# Descriptions here are intentionally minimal - the full decision rules live
# in ANALYSIS_INSTRUCTIONS (the verbatim protocol text) below, not here.
# Only the 7 fields assessable without source-code access are included;
# "Key concept code class" and "Recommended addition" require inspecting the
# project's source code, which isn't provided to the LLM in this pipeline.
# ============================================================
class RequirementAnalysis(BaseModel):
    syntactic_adherence: Literal["Yes", "No"] = Field(
        description=(
            "Syntactic adherence, per protocol criterion 1. "
            "LLM calibration: a requirement that reproduces most structural elements "
            "(participants, entry/exit conditions, flow of events) but omits the gold "
            "set's own use-case title/name heading is still a structural mismatch, not Yes."
        )
    )
    semantic_adherence: Literal["Yes", "No"] = Field(
        description=(
            "Semantic adherence, per protocol criterion 2. "
            "LLM calibration: a named actor triggering the action does not by itself make it "
            "user-centered - a raw existence or availability check (e.g. checking if a login or id "
            "already exists) with no further business outcome is still an internal operation, even "
            "though performing CRUD on a domain entity (creating, reading/retrieving, deleting, or "
            "updating a record) is not. Conversely, a requirement whose substance describes building, "
            "implementing, or managing a software component or module (e.g. 'develop/implement a system "
            "for...', referencing internal constructs like static variables, table models, or UI widgets) "
            "is an internal/development-task description, not user-centered, even if it carries incidental "
            "actor or flow-of-events scaffolding."
        )
    )
    key_concept_manually_matched_to: List[str] = Field(
        default_factory=list,
        description=(
            "Key concept manually matched to, per protocol criterion 4. "
            "LLM calibration: only include an ID when the shared functionality is clear and explicit, "
            "not merely plausible, adjacent, or inferred - sharing only a topic, domain, or affected "
            "data is not enough. A gold requirement whose scope only partially overlaps (e.g. it covers "
            "a broader report or listing that includes this data among other things) is not a match "
            "unless that broader scope is itself what the generated requirement covers. A generic "
            "validation, error-handling, or input-check step that could equally apply to many unrelated "
            "use cases (e.g. 'reject an invalid id') does not, by itself, match every gold requirement "
            "that happens to share that same generic step - only match it to a specific gold requirement "
            "when the generated requirement's primary functionality (not just an incidental validation "
            "step within it) corresponds to that gold requirement's use case. "
            "Unique matching gold standard IDs, or [] if unmatched."
        ),
    )
    secondary_concept_match: List[Literal["Yes", "Partly", "No"]] = Field(
        default_factory=list,
        description=(
            "Secondary concept match, per protocol criterion 5. "
            "LLM calibration: additional user actions include navigating to or viewing a list of items "
            "(e.g. as the result of a prior search/list use case) before selecting the target item - "
            "not just its named actors. A generated requirement that skips straight to the target item "
            "without any such prior list/search step is missing that action, not merely restating it "
            "differently. Do not treat routine, expected UI mechanics - displaying an entry/edit form, "
            "confirming before a destructive action, or showing a success/result message - as a missing "
            "secondary action; only a distinct, substantive interactive step beyond this kind of "
            "boilerplate counts. No requires clear absence or contradiction, not merely that an item "
            "isn't explicitly restated. Partly requires naming one specific shared item AND one specific "
            "missing/different item - it's the exception, not a default for uncertainty. Yes requires "
            "every item to be clearly shared. A different mechanism or wording for the same action is a "
            "details-match difference, not a missing concept; a generically-named actor group (e.g. "
            "'Citizen or Administrator or Employee') covers any one specifically-named matching actor. "
            "One label per matched ID, in the same order; [] if unmatched."
        ),
    )
    details_match: List[Literal["Yes", "Partly", "No"]] = Field(
        default_factory=list,
        description=(
            "Details match, per protocol criterion 6. "
            "LLM calibration: Partly requires nearly all specified details to be preserved, with only "
            "a single minor, trivial gap - if more than one specified detail is missing, generalized, or "
            "altered, it is No, not Partly. A vaguer or looser restatement of a specified detail counts "
            "as missing, not preserved. "
            "One label per matched ID, in the same order; [] if unmatched."
        ),
    )
    abstraction_level: Literal["Equal", "Lower", "Higher"] = Field(
        description=(
            "Abstraction level, per protocol criterion 7. "
            "LLM calibration: judge scope against the gold set's typical scope for this functionality, "
            "not whether the generated requirement reads as complete and well-formed on its own. A "
            "single action is Lower if the gold set typically groups it with related actions into a "
            "broader requirement, or if its matched gold requirement's own description covers a wider "
            "surrounding workflow (e.g. reaching this action via a prior list or search) that the "
            "generated requirement omits. For an unmatched requirement, judge scope against the gold "
            "set's overall typical scope: a terse single-action stub is still Lower even with no "
            "specific gold requirement to compare against."
        )
    )
    detail_level: Literal["Equal", "Lower", "Higher"] = Field(
        description=(
            "Detail Level, per protocol criterion 8. "
            "LLM calibration: Equal requires clearly matching, not merely approximating, the gold "
            "set's typical specificity - treat a missing gold-typical data field, format, or condition "
            "as evidence of Lower, even if the general action is still described. Generic implementation-style "
            "padding (numeric validation ranges, timestamps, logging, connection-error handling, generic "
            "confirmation/duplicate messages) does not substitute for that gold-specific content - a "
            "requirement can read as technically detailed while still being Lower if it swaps out the "
            "gold's specific named fields, formats, or thresholds for this kind of generic detail."
        )
    )


ANALYSIS_FIELDS = list(RequirementAnalysis.model_fields.keys())

# Batch schema: same fields as RequirementAnalysis, plus a requirement_id so the LLM
# can return one entry per requirement in a single response (used by --batch mode).
RequirementAnalysisItem = create_model(
    "RequirementAnalysisItem",
    requirement_id=(str, Field(description="The id of the requirement this analysis corresponds to, exactly as given in the input (e.g. 'code_req_42').")),
    **{name: (field.annotation, field) for name, field in RequirementAnalysis.model_fields.items()},
)


class BatchRequirementAnalysis(BaseModel):
    results: List[RequirementAnalysisItem] = Field(
        description="One analysis entry per generated requirement given, in the same order, with requirement_id matching the input id."
    )


# ============================================================
# Inter-rater agreement configuration (human sheet column -> LLM field name)
# ============================================================
HUMAN_TO_FIELD = {
    "Syntactic adherence": "syntactic_adherence",
    "Semantic adherence": "semantic_adherence",
    "Key concept manually matched to": "key_concept_manually_matched_to",
    "Secondary concept match": "secondary_concept_match",
    "Details match": "details_match",
    "Abstraction level": "abstraction_level",
    "Detail Level": "detail_level",
}

# Fields compared directly, one value per requirement
SIMPLE_AGREEMENT_FIELDS = [
    "syntactic_adherence",
    "semantic_adherence",
    "abstraction_level",
    "detail_level",
]

# Fields only meaningful per matched gold-standard id
PER_ID_AGREEMENT_FIELDS = ["secondary_concept_match", "details_match"]

PairList = List[Tuple[str, str]]


# ============================================================
# Analysis instructions - verbatim excerpt from the manual coding protocol.
# Sections "3 Key concept code class", "9 Recommended addition", "1 Purpose
# and preparation" and "4 Recording and review" are omitted: they either
# require source-code access not given to the LLM here, or are purely about
# human record-keeping/versioning and don't apply to a single LLM judgment.
# Table-formatted decision rules have been rendered as plain "Label: rule"
# lines; wording is otherwise unchanged from the protocol.
# ============================================================
ANALYSIS_INSTRUCTIONS = """
2 Coding procedure

Unit of analysis. One generated requirement is one record. Secondary-concept and detail-match
ratings apply separately to each generated-gold requirement pair. Many-to-many mappings are
permitted.

Read each generated requirement in full and identify its primary functionality or user goal.
Examine the full project gold set when mapping requirements.

Assess secondary concepts and details only for mapped pairs. Assess abstraction and detail level
for both matched and unmatched requirements.

3 Coding criteria

1 Syntactic adherence

Definition. Adherence to the presentation and syntactic format of the project's gold-standard
requirements.

Output and applicability. Yes or No for every generated requirement.

Decision rules. Record Yes when the same general structural pattern is clearly present, including
relevant headings, sentence structure and style. Equivalent heading names are acceptable. Record No
when the pattern is absent or substantially different. General readability alone does not establish
adherence.

Example. If the gold standard uses named use cases with actors and a flow, an equivalent structured
use case receives Yes; a plain prose sentence receives No. For a gold standard written as prose,
prose is not itself grounds for No.

2 Semantic adherence

Definition. Adherence to the main semantic focus of the project's gold-standard requirements, such
as user-centered goals.

Output and applicability. Yes or No for every generated requirement.

Decision rules. Record Yes when the main focus follows the gold standard's semantic style. For
user-centered gold requirements, a meaningful user goal qualifies, whereas an internal technical
operation alone receives No. A named actor does not automatically make an operation user-centered.
Technical steps within a user goal do not by themselves cause No.

Example. "The employee views citizen information" expresses a user goal. "The system loads the
citizen object into a session" describes an internal operation and should receive No. Missing data
formats are assessed under details rather than determining semantic adherence.

4 Key concept manually matched to

Definition. The gold-standard requirements sharing the generated requirement's primary
functionality or user goal. Examples of key concepts include login, user deletion and password
recovery.

Output and applicability. Unique gold-standard IDs, separated by commas, for every generated
requirement with a match. Record no match when none applies.

Decision rules. Examine the complete gold set for the project. Include every requirement with a
clearly shared key concept, allowing differences in wording, actors, abstraction and detail. Shared
terminology, affected data or supporting steps alone do not establish a match. A broad requirement
can map to several narrower requirements when the functionality it expresses clearly encompasses
their key concepts.

A requirement covering a stage of a larger goal can map when that stage directly expresses
functionality within the gold requirement. Record the narrower scope and missing actions under the
corresponding variables. Do not map a merely adjacent function.

Examples. "Search users" does not match "Delete users" solely because deletion includes a search.

5 Secondary concept match

Definition. Coverage of the mapped gold requirement's specified actors or roles and additional user
actions beyond the shared key concept.

Output and applicability. Yes, Partly or No for each mapped gold ID, in the same order as the ID
list. Not applicable for an unmatched requirement.

Yes: All specified secondary concepts are covered.
Partly: Some are covered and others are missing or different. Identify at least one shared item and
one missing or different item.
No: None of the specified secondary concepts are covered.

Decision rules. Compare actors and additional actions as discrete items. Equivalent wording counts.
An implied or generic role is not automatically equivalent to a specified role. Different mechanisms
for carrying out the same action do not alone make it a different secondary concept. Assess
prescribed mechanics, formats and validations under Details match.

Example. Gold: a librarian views and renews loans, with viewing as the shared key concept. Librarian
plus viewing and renewal receives Yes; librarian plus viewing alone receives Partly; a distinct actor
plus viewing alone receives No. Do not use Partly merely to express uncertainty.

6 Details match

Definition. Preservation of the mapped gold requirement's specified data fields, values, formats,
constraints, conditional behavior and prescribed steps.

Output and applicability. Yes, Partly or No for each mapped gold ID, in the same order as the ID
list. Not applicable for an unmatched requirement.

Yes: All specified details are explicitly preserved.
Partly: Nearly all details are preserved, with minor omissions or differences. Defining constraints
and required behavior remain intact.
No: Defining details are omitted, generalised or changed, or most specific information is absent.

Decision rules. Do not infer unstated details. A generic statement such as "validate input" does not
preserve a specified validation rule. Additional generated details do not compensate for missing
gold details.

Examples. If gold requires a six-digit numeric employee number, "enter an employee number" omits a
defining format and receives No. Preserving all defining rules while omitting a minor navigation
instruction can receive Partly.

7 Abstraction level

Definition. The breadth of the main functionality relative to the typical functional scope of the
project's gold-standard requirements.

Output and applicability. Equal, Lower or Higher for every generated requirement, whether matched or
unmatched.

Equal: Covers a complete goal or group of related actions of comparable scope.
Lower: Covers a narrower action, supporting operation or internal subtask.
Higher: Combines several goals or functions into a broader requirement.

Decision rules. Compare functional scope with the project baseline. Count distinct functionality
rather than fields, validation rules or flow steps. More implementation detail does not by itself
imply Higher abstraction.

Example. If a typical gold requirement groups viewing, editing and deleting user data, deletion
alone is Lower. Descriptions of internal system operations would be Lower. If the project usually
specifies those actions separately, deletion may be Equal. Combining several separately specified
goals into broader user management may be Higher.

8 Detail Level

Definition. The amount and specificity of concrete information relative to typical requirements in
the complete project gold set.

Output and applicability. Equal, Lower or Higher for every generated requirement, whether matched or
unmatched.

Equal: Provides a comparable amount and specificity of concrete information.
Lower: Provides fewer or less specific details about data, steps, conditions or outcomes.
Higher: Provides more extensive concrete information.

Decision rules. Assess the whole requirement against the project baseline. Concrete implementation
details count; text length, repeated information, generic steps and technical vocabulary alone do
not establish higher detail. Keep the amount of detail separate from whether those details match
gold.

Example. Where gold typically specifies the fields used to create a user, "create a user" has Lower
detail. A requirement naming comparable fields may be Equal. A requirement adding extensive concrete
validation and outcome rules may be Higher.

A requirement may have Higher detail and Details match = No.
"""


# ============================================================
# Load gold standard requirements for a dataset
# ============================================================
def load_gold_standard(excel_path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=f"{dataset}_gs", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df[["#id", "Requirement"]].dropna(subset=["#id"]).reset_index(drop=True)


def format_gold_standard(df_gs: pd.DataFrame) -> str:
    return "\n\n".join(f"{row['#id']}: {row['Requirement']}" for _, row in df_gs.iterrows())


# ============================================================
# Load the generated requirements for one dataset/variant sheet
# ============================================================
def load_variant_requirements(excel_path: Path, dataset: str, variant: str) -> pd.DataFrame:
    sheet = f"{dataset}_{variant}"
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    header = df_raw.iloc[1].astype(str).str.strip()
    df = df_raw.iloc[2:].copy()
    df.columns = header
    df = df.loc[:, ~(df.isna().all(axis=0))]
    df.columns = [str(c).strip() for c in df.columns]
    id_col = df.columns[0]
    req_col = "Requirement" if "Requirement" in df.columns else df.columns[1]
    df = df.dropna(subset=[id_col]).reset_index(drop=True)
    return df[[id_col, req_col]].rename(columns={id_col: "id", req_col: "requirement"})


# ============================================================
# Inter-rater agreement: load the human-authored sheet for one dataset/variant
# ============================================================
def load_human_analysis(excel_path: Path, dataset: str, variant: str) -> pd.DataFrame:
    sheet = f"{dataset}_{variant}"
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    header = df_raw.iloc[1].astype(str).str.strip()
    df = df_raw.iloc[2:].copy()
    df.columns = header
    df = df.loc[:, ~(df.isna().all(axis=0))]
    df.columns = [str(c).strip() for c in df.columns]
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "id", **HUMAN_TO_FIELD})
    df = df.dropna(subset=["id"]).reset_index(drop=True)
    df["id"] = df["id"].astype(str).str.strip()
    return df


def _norm(value) -> Optional[str]:
    """Normalize a cell value for comparison; returns None for empty/NA."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip().lower()
    if s in ("", "nan", "na", "n/a", "none", "-"):
        return None
    return s


def split_list(value) -> List[str]:
    if _norm(value) is None:
        return []
    return [tok.strip() for tok in str(value).split(",") if tok.strip()]


def normalize_id(value) -> str:
    return str(value).strip().upper()


def jaccard_similarity(a: frozenset, b: frozenset) -> float:
    """Intersection-over-union of two gold-id sets. Both empty is treated as
    perfect trivial agreement (1.0), since both raters agree nothing maps."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dice_f1(a: frozenset, b: frozenset) -> float:
    """Set-level F1 (Dice coefficient): 2|A n B| / (|A| + |B|)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


SetPairList = List[Tuple[frozenset, frozenset]]


def collect_agreement_pairs(
    dataset: str,
    variant: str,
    records: Dict[str, "PairList"],
    set_pairs: SetPairList,
    gold_ids: "frozenset[str]",
) -> int:
    """Compare the human sheet against the LLM checkpoint for one dataset/variant,
    appending (human, llm) value pairs to `records`, keyed by field name, and
    (human_set, llm_set) gold-id set pairs to `set_pairs`.
    Returns the number of requirements analyzed by both raters."""
    try:
        human_df = load_human_analysis(EXCEL_PATH, dataset, variant)
    except Exception as e:
        print(f"Skipping {dataset}_{variant} (human sheet): {e}")
        return 0

    llm_checkpoint = load_checkpoint(dataset, variant)
    if not llm_checkpoint:
        print(f"Skipping {dataset}_{variant}: no LLM checkpoint found.")
        return 0

    n_pairs = 0
    for _, h_row in human_df.iterrows():
        l_row = llm_checkpoint.get(h_row["id"])
        if l_row is None:
            continue
        n_pairs += 1

        for field in SIMPLE_AGREEMENT_FIELDS:
            h_val = _norm(h_row.get(field))
            l_val = _norm(l_row.get(field))
            if h_val is None or l_val is None:
                continue
            records.setdefault(field, []).append((h_val, l_val))

        h_ids = split_list(h_row.get("key_concept_manually_matched_to"))
        l_ids = split_list(l_row.get("key_concept_manually_matched_to"))
        records.setdefault("is_mapped_to_gold", []).append(
            ("yes" if h_ids else "no", "yes" if l_ids else "no")
        )
        set_pairs.append((frozenset(h_ids), frozenset(l_ids)))

        # Strict per-gold-id binary agreement: for every gold-standard id, did each
        # rater include it in this requirement's matched-to set? Flattening this
        # into yes/no pairs (one per gold id) lets Cohen's kappa be computed over
        # the full binary indicator matrix, exactly-matching-ids-as-true style.
        h_ids_norm = {normalize_id(i) for i in h_ids}
        l_ids_norm = {normalize_id(i) for i in l_ids}
        for gold_id in gold_ids:
            records.setdefault("key_concept_match_strict", []).append((
                "yes" if gold_id in h_ids_norm else "no",
                "yes" if gold_id in l_ids_norm else "no",
            ))

        # Per-gold-id fields (secondary concept / details match) are only meaningful
        # for gold ids both raters mapped this requirement to.
        for field in PER_ID_AGREEMENT_FIELDS:
            h_map = dict(zip(h_ids, split_list(h_row.get(field))))
            l_map = dict(zip(l_ids, split_list(l_row.get(field))))
            for gold_id in set(h_map) & set(l_map):
                h_val, l_val = _norm(h_map[gold_id]), _norm(l_map[gold_id])
                if h_val is None or l_val is None:
                    continue
                records.setdefault(field, []).append((h_val, l_val))

    return n_pairs


def kappa_interpretation(k: float) -> str:
    # Landis & Koch (1977) rule of thumb.
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def summarize_agreement(records: Dict[str, "PairList"]) -> pd.DataFrame:
    rows = []
    for field, pairs in records.items():
        if not pairs:
            continue
        human_vals, llm_vals = zip(*pairs)
        n = len(pairs)
        percent_agreement = sum(h == l for h, l in pairs) / n * 100
        labels = sorted(set(human_vals) | set(llm_vals))
        kappa = cohen_kappa_score(human_vals, llm_vals, labels=labels) if len(labels) >= 2 else float("nan")
        rows.append({
            "field": field,
            "n_pairs": n,
            "percent_agreement": round(percent_agreement, 1),
            "cohens_kappa": round(kappa, 3) if kappa == kappa else None,
            "interpretation": kappa_interpretation(kappa) if kappa == kappa else "n/a",
        })
    return pd.DataFrame(rows)


def summarize_set_agreement(set_pairs: SetPairList) -> pd.DataFrame:
    """Cohen's kappa doesn't apply to multi-label sets like
    key_concept_manually_matched_to (there's no fixed small label space to build
    a confusion matrix from). Instead, report the standard set-agreement metrics:
    mean Jaccard index (IoU) and mean Dice/set-F1, per requirement, macro-averaged.
    Both-empty (neither rater mapped the requirement to any gold id) counts as
    trivial perfect agreement (1.0)."""
    n = len(set_pairs)
    if n == 0:
        return pd.DataFrame()
    exact_match_pct = sum(h == l for h, l in set_pairs) / n * 100
    mean_jaccard = sum(jaccard_similarity(h, l) for h, l in set_pairs) / n
    mean_dice_f1 = sum(dice_f1(h, l) for h, l in set_pairs) / n
    return pd.DataFrame([{
        "field": "key_concept_manually_matched_to",
        "n_pairs": n,
        "exact_set_match_pct": round(exact_match_pct, 1),
        "mean_jaccard": round(mean_jaccard, 3),
        "mean_dice_f1": round(mean_dice_f1, 3),
    }])


AGREEMENT_SHEET_NAME = "Agreement"


def _write_titled_table(ws, title: str, df: pd.DataFrame, start_row: int) -> int:
    """Writes `title` as a bold header at `start_row` (1-indexed openpyxl row) and
    the dataframe starting the row after. Returns the next free row (1-indexed)."""
    cell = ws.cell(row=start_row, column=1, value=title)
    cell.font = Font(bold=True)
    row = start_row + 1
    for values in dataframe_to_rows(df, index=False, header=True):
        for col, value in enumerate(values, start=1):
            ws.cell(row=row, column=col, value=value)
        row += 1
    return row + 1  # blank gap row


def write_agreement_to_workbook(results: Dict[str, Optional[pd.DataFrame]]) -> None:
    """Writes all inter-rater agreement tables onto a single 'Agreement' sheet, placed
    first, in Manual_analysis_llm_protocol.xlsx, so everything is in one place instead
    of scattered across separate CSV files. Written directly with openpyxl (rather than
    repeated pandas to_excel calls) so that writing one table doesn't wipe out the
    tables already written to the same sheet."""
    wb = load_workbook(OUTPUT_XLSX) if OUTPUT_XLSX.exists() else Workbook()
    if AGREEMENT_SHEET_NAME in wb.sheetnames:
        del wb[AGREEMENT_SHEET_NAME]
    if wb.sheetnames == ["Sheet"] and wb["Sheet"].max_row == 1 and wb["Sheet"].max_column == 1 and wb["Sheet"]["A1"].value is None:
        # Freshly created workbook still has the default empty sheet; drop it.
        del wb["Sheet"]
    ws = wb.create_sheet(AGREEMENT_SHEET_NAME, 0)

    row = 1
    row = _write_titled_table(ws, "Overall inter-rater agreement (human vs LLM)", results["summary"], row)
    if results.get("set_summary") is not None and not results["set_summary"].empty:
        row = _write_titled_table(ws, "Key concept match agreement (set overlap, human vs LLM)", results["set_summary"], row)
    if results.get("variant_summary") is not None:
        row = _write_titled_table(ws, "Per-variant inter-rater agreement", results["variant_summary"], row)
    if results.get("variant_set_summary") is not None:
        row = _write_titled_table(ws, "Per-variant key concept match agreement (set overlap)", results["variant_set_summary"], row)

    wb.save(OUTPUT_XLSX)
    print(f"\nWrote agreement results to the '{AGREEMENT_SHEET_NAME}' sheet in {OUTPUT_XLSX}")


def compute_agreement(datasets: List[str], variants: List[str], per_variant: bool = False) -> None:
    overall_records: Dict[str, "PairList"] = {}
    per_variant_records: Dict[str, Dict[str, "PairList"]] = {}
    overall_set_pairs: SetPairList = []
    per_variant_set_pairs: Dict[str, SetPairList] = {}
    total_pairs = 0

    for dataset in datasets:
        try:
            gold_ids = frozenset(normalize_id(i) for i in load_gold_standard(EXCEL_PATH, dataset)["#id"])
        except Exception as e:
            print(f"Skipping {dataset}: failed to load gold standard ids: {e}")
            continue
        for variant in variants:
            local_records: Dict[str, "PairList"] = {}
            local_set_pairs: SetPairList = []
            n = collect_agreement_pairs(dataset, variant, local_records, local_set_pairs, gold_ids)
            total_pairs += n
            for field, pairs in local_records.items():
                overall_records.setdefault(field, []).extend(pairs)
                per_variant_records.setdefault(variant, {}).setdefault(field, []).extend(pairs)
            overall_set_pairs.extend(local_set_pairs)
            per_variant_set_pairs.setdefault(variant, []).extend(local_set_pairs)

    print(f"\nMatched {total_pairs} requirements with both a human and LLM analysis.\n")

    summary = summarize_agreement(overall_records)
    print("=== Overall inter-rater agreement (human vs LLM) ===")
    print(summary.to_string(index=False))

    set_summary = summarize_set_agreement(overall_set_pairs)
    if not set_summary.empty:
        print("\n=== Key concept match agreement (set overlap, human vs LLM) ===")
        print(set_summary.to_string(index=False))

    variant_summary = None
    variant_set_summary = None
    if per_variant:
        rows = []
        for variant, records in per_variant_records.items():
            df = summarize_agreement(records)
            df.insert(0, "variant", variant)
            rows.append(df)
        if rows:
            variant_summary = pd.concat(rows, ignore_index=True)
            print("\n=== Per-variant inter-rater agreement ===")
            print(variant_summary.to_string(index=False))

        set_rows = []
        for variant, pairs in per_variant_set_pairs.items():
            df = summarize_set_agreement(pairs)
            if df.empty:
                continue
            df.insert(0, "variant", variant)
            set_rows.append(df)
        if set_rows:
            variant_set_summary = pd.concat(set_rows, ignore_index=True)
            print("\n=== Per-variant key concept match agreement (set overlap) ===")
            print(variant_set_summary.to_string(index=False))

    write_agreement_to_workbook({
        "summary": summary,
        "set_summary": set_summary,
        "variant_summary": variant_summary,
        "variant_set_summary": variant_set_summary,
    })


# ============================================================
# Prompting
# ============================================================
def build_system_prompt(instructions: str, gold_standard: str) -> str:
    return f"""You are performing an analysis of an automatically generated software \
requirement, comparing it against a gold standard set of requirements for the same system, \
following the manual coding protocol excerpt below exactly.

{instructions}

Calibration note: when a rating is ambiguous between two options (e.g. Equal vs Lower, or No vs \
Partly), choose the stricter, more conservative option unless the requirement text gives clear, \
explicit support for the more lenient one. Do not assume typical or default behavior that isn't \
actually stated in the requirement.

Here are the GOLD STANDARD requirements for this system:
{gold_standard}

Analyze the generated requirement you are given and return your assessment for \
ALL of the properties above, following the protocol's decision rules exactly."""


def build_user_prompt(requirement_id: str, requirement_text: str) -> str:
    return f"Generated requirement {requirement_id}: {requirement_text}"


def analyze_requirement(
    client: OpenAI,
    system_prompt: str,
    requirement_id: str,
    requirement_text: str,
) -> Optional[RequirementAnalysis]:
    user_prompt = build_user_prompt(requirement_id, requirement_text)
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=RequirementAnalysis,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"  LLM call failed for {requirement_id} (attempt {attempt}/{RETRIES}): {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
    return None


def build_batch_user_prompt(requirements: List[Tuple[str, str]]) -> str:
    parts = [f"Generated requirement {req_id}: {req_text}" for req_id, req_text in requirements]
    return (
        "You are given the following generated requirements to analyze. Return exactly "
        "one result per requirement, in the same order, with requirement_id set to the "
        "id given below (e.g. 'code_req_42').\n\n" + "\n\n".join(parts)
    )


def analyze_requirements_batch(
    client: OpenAI,
    system_prompt: str,
    requirements: List[Tuple[str, str]],
) -> Dict[str, RequirementAnalysisItem]:
    """Analyzes a whole batch (e.g. all pending requirements for a dataset/variant) in
    a single LLM call. Returns a dict keyed by requirement_id."""
    user_prompt = build_batch_user_prompt(requirements)
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=BatchRequirementAnalysis,
            )
            parsed = response.choices[0].message.parsed
            return {item.requirement_id: item for item in parsed.results}
        except Exception as e:
            print(f"  Batch LLM call failed (attempt {attempt}/{RETRIES}): {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
    return {}


# ============================================================
# Checkpointing (resume support)
# ============================================================
def serialize_result(result: RequirementAnalysis) -> Dict[str, str]:
    """Flatten the (possibly list-valued) analysis fields into comma-joined strings,
    matching the format used in the human-authored sheets (e.g. 'EA1, EA2')."""
    data = result.model_dump()
    flat = {}
    for field in ANALYSIS_FIELDS:
        value = data[field]
        if isinstance(value, list):
            flat[field] = ", ".join(str(v) for v in value)
        else:
            flat[field] = value
    return flat


def checkpoint_path(dataset: str, variant: str) -> Path:
    return CHECKPOINT_DIR / f"{dataset}_{variant}.csv"

def load_checkpoint(dataset: str, variant: str) -> Dict[str, dict]:
    path = checkpoint_path(dataset, variant)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return {row["id"]: row.to_dict() for _, row in df.iterrows()}


def append_checkpoint_row(dataset: str, variant: str, row: dict) -> None:
    path = checkpoint_path(dataset, variant)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "requirement"] + ANALYSIS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Processing one dataset/variant sheet
# ============================================================
def process_variant_batch(
    client: OpenAI,
    dataset: str,
    variant: str,
    df_reqs: pd.DataFrame,
    system_prompt: str,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Like process_variant, but sends multiple requirements to the LLM per call
    instead of one at a time. By default all pending requirements for this
    dataset/variant are sent in a single call; pass batch_size to chunk them."""
    already_done = load_checkpoint(dataset, variant)
    pending = [
        (str(req.id).strip(), str(req.requirement))
        for req in df_reqs.itertuples(index=False)
        if str(req.id).strip() not in already_done
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"[{dataset}/{variant}] nothing to do (batch mode).")
        return

    chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)] if batch_size else [pending]
    for chunk in chunks:
        print(f"[{dataset}/{variant}] analyzing batch of {len(chunk)} requirements...")
        results_by_id = analyze_requirements_batch(client, system_prompt, chunk)
        if not results_by_id:
            print(f"  Batch call failed after {RETRIES} attempts, skipping this batch.")
            continue
        for req_id, req_text in chunk:
            result = results_by_id.get(req_id)
            if result is None:
                print(f"  Missing result for {req_id} in batch response, skipping.")
                continue
            row = {"id": req_id, "requirement": req_text, **serialize_result(result)}
            append_checkpoint_row(dataset, variant, row)


# ============================================================
# Compile all checkpoints into the final output workbook
# ============================================================
def compile_output_workbook() -> None:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for dataset in DATASETS:
            for variant in VARIANTS:
                path = checkpoint_path(dataset, variant)
                if not path.exists():
                    continue
                df = pd.read_csv(path, dtype=str)
                sheet_name = f"{dataset}_{variant}"[:31]
                # Match the human sheet layout: title row, then header, then data.
                title_row = pd.DataFrame([[f"llm_generated_{dataset}_{variant}"] + [None] * (len(df.columns) - 1)],
                                          columns=df.columns)
                out_df = pd.concat([title_row, df], ignore_index=True)
                out_df.to_excel(writer, sheet_name=sheet_name, header=True, index=False)
    print(f"Wrote combined results to {OUTPUT_XLSX}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run automated (LLM-based) manual analysis using the verbatim coding protocol as instructions.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Subset of datasets to process.")
    parser.add_argument("--variants", nargs="*", default=VARIANTS, help="Subset of variants to process.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of new requirements per variant (for testing).")
    parser.add_argument("--compile-only", action="store_true", help="Skip LLM calls, just rebuild the output workbook from checkpoints.")
    parser.add_argument("--batch-size", type=int, default=None, help="Max number of requirements per LLM call (default: all pending requirements in one call).")
    parser.add_argument("--agreement", action="store_true", help="After running (or with --compile-only, using existing checkpoints), also compute Cohen's kappa between the human and LLM analyses.")
    parser.add_argument("--agreement-only", action="store_true", help="Skip the LLM pipeline entirely and only compute agreement from existing checkpoints.")
    parser.add_argument("--per-variant", action="store_true", help="With --agreement, also break down agreement by variant.")
    args = parser.parse_args()

    if args.agreement_only:
        compile_output_workbook()
        compute_agreement(args.datasets, args.variants, per_variant=args.per_variant)
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.compile_only:
        client = get_client()

        for dataset in args.datasets:
            print(f"=== Loading context for {dataset} ===")
            df_gs = load_gold_standard(EXCEL_PATH, dataset)
            gold_standard_text = format_gold_standard(df_gs)
            system_prompt = build_system_prompt(ANALYSIS_INSTRUCTIONS, gold_standard_text)

            for variant in args.variants:
                try:
                    df_reqs = load_variant_requirements(EXCEL_PATH, dataset, variant)
                except Exception as e:
                    print(f"Failed to load sheet {dataset}_{variant}: {e}")
                    continue
                process_variant_batch(client, dataset, variant, df_reqs, system_prompt, limit=args.limit, batch_size=args.batch_size)

    compile_output_workbook()

    if args.agreement:
        compute_agreement(args.datasets, args.variants, per_variant=args.per_variant)


if __name__ == "__main__":
    main()
