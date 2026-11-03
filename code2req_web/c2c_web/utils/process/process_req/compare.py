import csv
import re
from pathlib import Path
from typing import Dict

from c2c_web.models import EvaluationFile, EvaluationUnit
from c2c_web.utils.utils import embed, cosine_similarity, get_all_from_vectordb, query_graphdb
import statistics
import pandas as pd
import matplotlib.pyplot as plt

def f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1, precision, recall

def extract_name(value):
    match = re.search(r'#(.*?)(?:_[a-f0-9]{10,}|)_feature$', value)
    if match:
        ex = match.group(1)
        return ex.replace("_"," ")

def get_embeddings(text_dict):
    return {identifier: embed(text)[0] for identifier, text in text_dict.items()}


def get_unit_dict(eval_file):
    return {
        unit.identifier: unit.description
        for unit in EvaluationUnit.objects.filter(file=eval_file)
    }


def _variant_specs(project):
    variants = []
    if project.use_feature:
        variants.append(("Code_Features", "generated_reqs_features.tsv"))
    if project.use_code:
        variants.append(("Req_from_code", "generated_reqs_code.tsv"))
    if project.use_rag and project.use_feature:
        variants.append(("Code_Features_RAG", "generated_reqs_features_rag.tsv"))
    if project.use_rag and project.use_code:
        variants.append(("Req_from_code_RAG", "generated_reqs_code_rag.tsv"))
    return variants


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("#id", "Requirement", "linked"))
        writer.writerows(rows)


def _reference_report_path(report_dir, reference_file):
    if reference_file.name == "Requirements":
        return report_dir / "reqs_linked.tsv"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", reference_file.name).strip("_.")
    return report_dir / f"reference_{reference_file.id}_{safe_name}_linked.tsv"


def compare(project, comparison_report_path, similarity_cutoff=0.8):
    report_dir = Path(comparison_report_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_names = {"Code_Features", "Req_from_code", "Code_Features_RAG", "Req_from_code_RAG"}
    reference_files = [
        file for file in project.evaluation_files.all()
        if file.name not in generated_names
    ]
    if not reference_files:
        return

    variants = _variant_specs(project)
    variant_files = {}
    for name, report_name in variants:
        generated_file = EvaluationFile.objects.filter(project=project, name=name).first()
        if generated_file is None:
            raise ValueError(
                f"Enabled requirement generator '{name}' did not create its output record. "
                "Check generation logs before comparing results."
            )
        variant_files[name] = {
            "file": generated_file,
            "report": report_name,
            "units": list(EvaluationUnit.objects.filter(file=generated_file)),
        }

    reference_units = {}
    for reference_file in reference_files:
        units = list(EvaluationUnit.objects.filter(file=reference_file))
        reference_units[reference_file.id] = units
        for unit in units:
            unit.linked_feature_ids = []
            unit.scores = []
            unit.save(update_fields=["linked_feature_ids", "scores"])

    for variant in variant_files.values():
        for unit in variant["units"]:
            unit.linked_feature_ids = []
            unit.scores = []
            unit.save(update_fields=["linked_feature_ids", "scores"])

    if not variants:
        for reference_file in reference_files:
            units = reference_units[reference_file.id]
            reference_file.f1_score = 0.0
            reference_file.num_linked_to_feature = 0
            reference_file.num_units = len(units)
            reference_file.save(update_fields=["f1_score", "num_linked_to_feature", "num_units"])
            _write_tsv(
                _reference_report_path(report_dir, reference_file),
                [(unit.identifier, unit.description.replace("\n", " "), "[]") for unit in units],
            )
        return

    variant_embeddings = {}
    for name, variant in variant_files.items():
        descriptions = {unit.identifier: unit.description for unit in variant["units"]}
        variant_embeddings[name] = get_embeddings(descriptions)

    reference_embeddings = {}
    for reference_file in reference_files:
        descriptions = {
            f"{reference_file.id}:{unit.identifier}": unit.description
            for unit in reference_units[reference_file.id]
        }
        reference_embeddings[reference_file.id] = get_embeddings(descriptions)

    linked_candidates_by_reference = {file.id: set() for file in reference_files}
    linked_references_by_reference = {file.id: set() for file in reference_files}
    linked_references_by_variant = {name: set() for name in variant_files}

    for reference_file in reference_files:
        units = reference_units[reference_file.id]
        embeddings = reference_embeddings[reference_file.id]
        for variant_name, variant in variant_files.items():
            candidates = variant["units"]
            candidate_embeddings = variant_embeddings[variant_name]
            for candidate in candidates:
                for reference_unit in units:
                    reference_key = f"{reference_file.id}:{reference_unit.identifier}"
                    similarity = cosine_similarity(
                        candidate_embeddings[candidate.identifier], embeddings[reference_key]
                    )
                    if similarity < similarity_cutoff:
                        continue

                    candidate.linked_feature_ids.append(reference_key)
                    candidate.scores.append(f"{similarity:.2f}")
                    reference_unit.linked_feature_ids.append(f"{variant_name}:{candidate.identifier}")
                    reference_unit.scores.append(f"{similarity:.2f}")
                    linked_candidates_by_reference[reference_file.id].add((variant_name, candidate.pk))
                    linked_references_by_reference[reference_file.id].add(reference_key)
                    linked_references_by_variant[variant_name].add(reference_key)

    total_reference_count = sum(len(units) for units in reference_units.values())
    for variant_name, variant in variant_files.items():
        candidates = variant["units"]
        linked_candidate_count = sum(bool(unit.linked_feature_ids) for unit in candidates)
        false_negative_count = total_reference_count - len(linked_references_by_variant[variant_name])
        score, _, _ = f1_score(
            linked_candidate_count,
            len(candidates) - linked_candidate_count,
            false_negative_count,
        )
        generated_file = variant["file"]
        generated_file.f1_score = score
        generated_file.num_linked_to_feature = linked_candidate_count
        generated_file.num_units = len(candidates)
        generated_file.save(update_fields=["f1_score", "num_linked_to_feature", "num_units"])
        _write_tsv(
            report_dir / variant["report"],
            [
                (unit.identifier, unit.description.replace("\n", " "), str(unit.linked_feature_ids))
                for unit in candidates
            ],
        )
        for unit in candidates:
            unit.save(update_fields=["linked_feature_ids", "scores"])

    total_candidate_count = sum(len(variant["units"]) for variant in variant_files.values())
    for reference_file in reference_files:
        units = reference_units[reference_file.id]
        linked_candidates = linked_candidates_by_reference[reference_file.id]
        linked_references = linked_references_by_reference[reference_file.id]
        score, _, _ = f1_score(
            len(linked_candidates),
            total_candidate_count - len(linked_candidates),
            len(units) - len(linked_references),
        )
        reference_file.f1_score = score
        reference_file.num_linked_to_feature = len(linked_references)
        reference_file.num_units = len(units)
        reference_file.save(update_fields=["f1_score", "num_linked_to_feature", "num_units"])
        _write_tsv(
            _reference_report_path(report_dir, reference_file),
            [
                (unit.identifier, unit.description.replace("\n", " "), str(unit.linked_feature_ids))
                for unit in units
            ],
        )
        for unit in units:
            unit.save(update_fields=["linked_feature_ids", "scores"])


