"""Validate cross-manuscript claims and publication hygiene for Jigsaw."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPERS = (
    ROOT / "paper" / "samplepaper.tex",
    ROOT / "paper" / "jigsaw_log2026.tex",
    ROOT / "paper" / "jigsaw_ecmlpkdd.tex",
)
OPENREVIEW_METADATA = ROOT / "paper" / "openreview_submission_metadata.md"

VENDOR_SPECIFIC_AI_NAMES = ("OpenAI Codex", "ChatGPT")

REQUIRED_CLAIMS = (
    "2,400",
    "1,800",
    "600",
    "7,200",
    "5,400",
    "14,400",
    "288",
    "88.6",
    "94.2",
    "92.8",
    "87.7",
    "31.9",
    "32.7",
    "89.7",
    "33.3",
    "39.2",
    "99.3",
    "100.0",
    "88.9",
    "44.4",
    "94.4",
    "86.1",
)

OPERATOR_TABLE_CLAIMS = ("150.6K", "2,815")

BANNED_TEXT = (
    "query_payload_v1",
    r"query\_payload\_v1",
    "mag_rgcn_best",
    r"mag\_rgcn\_best",
    "mag_rgcn_final",
    r"mag\_rgcn\_final",
    "continuation/v7",
    "vertexlabelledlad",
    r"CPU\_X\_8",
    "release CSVs",
    "audited artifacts",
    "--overlap-max-parts",
    "--overlap-min-support",
    "--overlap-label-compatible",
    "candidate is the whole graph",
    "The breadth of one-hop overlap is not buying recall",
    "selective-overlap fix",
    "recall-preserving selective overlap",
    "support-filtered boundary-overlap",
    "support-filtered boundary overlap",
    "not yet a recall-preservation result",
    "not recall preservation",
    "production-matrix totals also include retrieval",
    "production-matrix per-query totals of Table~\\ref{tab:production_matrix} additionally include retrieval time",
    "retrieval is a prerequisite at scale",
    "end-to-end runtime",
    "Total s",
    "64.0K",
    "49.9K",
    "7,081",
    "7{,}081",
    r"${\approx}25$\,s",
    "every one exhausting the time budget",
    "$e=128$",
)


def normalized(text: str) -> str:
    return text.replace("{,}", ",")


def command_keys(text: str, command: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(rf"\\{command}(?:\[[^\]]*\])?\{{([^}}]+)\}}", text):
        keys.extend(part.strip() for part in match.group(1).split(","))
    return keys


def validate_structure(path: Path, text: str, errors: list[str]) -> None:
    labels = command_keys(text, "label")
    refs = command_keys(text, "ref") + command_keys(text, "pageref")
    citations = command_keys(text, "cite")
    bibitems = command_keys(text, "bibitem")

    duplicate_labels = sorted(key for key, count in Counter(labels).items() if count > 1)
    duplicate_bibitems = sorted(
        key for key, count in Counter(bibitems).items() if count > 1
    )
    undefined_refs = sorted(set(refs) - set(labels))
    undefined_citations = sorted(set(citations) - set(bibitems))

    for label, values in (
        ("duplicate labels", duplicate_labels),
        ("duplicate bibliography keys", duplicate_bibitems),
        ("undefined references", undefined_refs),
        ("undefined citations", undefined_citations),
    ):
        if values:
            errors.append(f"{path.name}: {label}: {values}")


def validate_claims(path: Path, text: str, errors: list[str]) -> None:
    plain = normalized(text)
    for claim in REQUIRED_CLAIMS:
        if claim not in plain:
            errors.append(f"{path.name}: required claim missing: {claim}")
    if path.name != "jigsaw_ecmlpkdd.tex":
        for claim in OPERATOR_TABLE_CLAIMS:
            if claim not in plain:
                errors.append(f"{path.name}: operator-table claim missing: {claim}")
    for banned in BANNED_TEXT:
        if banned in text:
            errors.append(f"{path.name}: publication-hygiene violation: {banned}")
    if "Cascade s" not in text:
        errors.append(f"{path.name}: production table must label Cascade s")
    if "positive-query solver timeouts" not in text:
        errors.append(f"{path.name}: production-table timeout scope is ambiguous")
    for citation in ("ref_graphsage", "ref_rgcn"):
        if citation not in command_keys(text, "cite"):
            errors.append(f"{path.name}: architecture citation missing: {citation}")
    if "$n{=}400$ locked K-hop queries" not in text:
        errors.append(f"{path.name}: FullCov ablation denominator/checkpoint scope missing")
    if "48 queries per dataset" not in text or "36 planted positives" not in text:
        errors.append(f"{path.name}: operator-ablation denominator missing")
    if "88.9{\\to}44.4" not in text and "88.9\\%$ to $44.4" not in text:
        errors.append(f"{path.name}: MAG overlap effect is not interpreted")
    if "94.4{\\to}86.1" not in text and "94.4\\%$ to $86.1" not in text:
        errors.append(f"{path.name}: Arxiv overlap effect is not interpreted")
    if path.name != "jigsaw_ecmlpkdd.tex":
        denominator_phrases = (
            "each displayed family has $n{=}300$",
            "Each displayed spatial family has $n{=}300$",
        )
        if not any(phrase in text for phrase in denominator_phrases):
            errors.append(f"{path.name}: retrieval-probe family denominator is ambiguous")


def validate_publication_style(path: Path, text: str, errors: list[str]) -> None:
    for vendor_name in VENDOR_SPECIFIC_AI_NAMES:
        if vendor_name.casefold() in text.casefold():
            errors.append(
                f"{path.name}: vendor-specific AI tool name must not appear: {vendor_name}"
            )

    if path.name != "jigsaw_log2026.tex":
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        if "&" in line or line.lstrip().startswith("%"):
            continue
        if "---" in line or " -- " in line or "\N{EM DASH}" in line:
            errors.append(
                f"{path.name}:{line_number}: dash-heavy prose is not permitted"
            )


def validate_matched_costs(path: Path, text: str, errors: list[str]) -> None:
    label = r"\label{tab:production_matrix}"
    start = text.find(label)
    if start < 0:
        errors.append(f"{path.name}: production table label is missing")
        return
    end = text.find(r"\end{table}", start)
    if end < 0:
        errors.append(f"{path.name}: production table is not closed")
        return
    production_table = text[start:end]
    row_pattern = re.compile(
        r"^(Cora|Arxiv)\s*&\s*([^&]+?)\s*&\s*"
        r"[^&]+&[^&]+&[^&]+&[^&]+&\s*([^&]+)&\s*([^&]+)&\s*([^\\]+)\\\\",
        re.MULTILINE,
    )
    rows = row_pattern.findall(production_table)
    if len(rows) != 12:
        errors.append(f"{path.name}: expected 12 Cora/Arxiv production rows, found {len(rows)}")
        return
    for dataset, method, candidate, cascade, solver in rows:
        if "--" in (candidate + cascade + solver):
            errors.append(f"{path.name}: matched cost still missing for {dataset}/{method.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-matched-costs", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    for path in PAPERS:
        text = path.read_text(encoding="utf-8")
        validate_structure(path, text, errors)
        validate_claims(path, text, errors)
        validate_publication_style(path, text, errors)
        if args.require_matched_costs:
            validate_matched_costs(path, text, errors)

    log_text = (ROOT / "paper" / "jigsaw_log2026.tex").read_text(encoding="utf-8")
    if not re.search(
        r"\\end\{thebibliography\}\s*(?:\}\s*)?\\clearpage\s*\\appendix",
        log_text,
    ):
        errors.append("jigsaw_log2026.tex: appendix must begin on a fresh page")

    metadata = OPENREVIEW_METADATA.read_text(encoding="utf-8")
    validate_publication_style(OPENREVIEW_METADATA, metadata, errors)
    for required in ("0/15", "88.6%", "99.3-100.0%", "2.4 GB", "10.2 GB"):
        if required not in metadata:
            errors.append(f"OpenReview metadata missing: {required}")
    for banned in (
        "candidate domain is the whole graph, which is intractable",
        "recall-preserving selective overlap",
        "99.7-100.0%",
    ):
        if banned in metadata:
            errors.append(f"OpenReview metadata contains stale claim: {banned}")

    if errors:
        raise SystemExit("MANUSCRIPT_VALIDATION_FAILED\n" + "\n".join(errors))
    print(f"MANUSCRIPT_VALIDATION_OK papers={len(PAPERS)}")


if __name__ == "__main__":
    main()
