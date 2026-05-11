"""
prepare_submission.py — Convert a participant .zip submission into the JSONL
format expected by runner.py.

Participant zip structure
-------------------------
    <team_name>.zip
    └── <team_name>/
        ├── <language>/
        │   ├── run1/
        │   │   ├── multiclinsum2_test_1_<language>_sum.txt
        │   │   ├── multiclinsum2_test_2_<language>_sum.txt
        │   │   └── ...
        │   ├── run2/   (optional)
        │   └── run3/   (optional)
        ├── <language>/
        │   └── run1/
        │       └── ...
        └── ...

Reference test-set structure
-----------------------------
    <reference_dir>/
    └── <language>/
        ├── multiclinsum2_test_1_<language>.txt       ← full clinical case
        ├── multiclinsum2_test_1_<language>_sum.txt   ← gold reference summary
        ├── multiclinsum2_test_2_<language>.txt
        ├── multiclinsum2_test_2_<language>_sum.txt
        └── ...

Output structure
----------------
    <output_dir>/<team_name>/<language>/<run>/results.jsonl

Each run gets its own results.jsonl.  Each line:
    {
        "case_id":            "multiclinsum2_test_1_ca",
        "language":           "ca",
        "full_case":          "<full clinical case text>",
        "reference_summary":  "<gold reference summary text>",
        "generated_summary":  "<participant generated summary text>"
    }

Usage
-----
    python prepare_submission.py \\
        --submission submissions/team_x.zip \\
        --reference_dir ../data/multiclinsum2_test_set \\
        --output_dir inputs/

    # Process only specific languages
    python prepare_submission.py \\
        --submission submissions/team_x.zip \\
        --reference_dir ../data/multiclinsum2_test_set \\
        --output_dir inputs/ \\
        --languages ca es en

    # Process only run1 (skip run2, run3)
    python prepare_submission.py \\
        --submission submissions/team_x.zip \\
        --reference_dir ../data/multiclinsum2_test_set \\
        --output_dir inputs/ \\
        --runs run1
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    """Read a text file, stripping leading/trailing whitespace."""
    return path.read_text(encoding="utf-8").strip()


def _case_id_from_sum_filename(filename: str) -> str:
    """
    Derive case_id from a participant summary filename.

    Examples
    --------
    "multiclinsum2_test_1_ca_sum.txt"  →  "multiclinsum2_test_1_ca"
    "multiclinsum2_test_12_es_sum.txt" →  "multiclinsum2_test_12_es"
    """
    if not filename.endswith("_sum.txt"):
        raise ValueError(
            f"Expected filename ending in '_sum.txt', got: '{filename}'"
        )
    return filename[: -len("_sum.txt")]


def convert_language(
    language: str,
    generated_summaries: dict[str, str],
    reference_dir: Path,
) -> tuple[list[dict], list[str]]:
    """
    Build JSONL records for one language.

    Parameters
    ----------
    language : str
        ISO 639-1 language code.
    generated_summaries : dict[str, str]
        Mapping of case_id → generated summary text (from the zip).
    reference_dir : Path
        Root of the test-set directory tree.

    Returns
    -------
    records : list[dict]
        JSONL-ready dicts, one per matched case.
    warnings : list[str]
        Human-readable warnings for missing/unmatched files.
    """
    lang_ref_dir = reference_dir / language
    records: list[dict] = []
    warnings: list[str] = []

    for case_id, generated in sorted(generated_summaries.items()):
        full_case_path = lang_ref_dir / f"{case_id}.txt"
        gold_summary_path = lang_ref_dir / f"{case_id}_sum.txt"

        if not full_case_path.exists():
            warnings.append(
                f"[{language}] Full case not found in reference: {full_case_path}"
            )
            continue

        if not gold_summary_path.exists():
            warnings.append(
                f"[{language}] Gold summary not found in reference: {gold_summary_path}"
            )
            continue

        records.append({
            "case_id": case_id,
            "language": language,
            "full_case": _read_txt(full_case_path),
            "reference_summary": _read_txt(gold_summary_path),
            "generated_summary": generated,
        })

    return records, warnings


def prepare_submission(
    zip_path: Path,
    reference_dir: Path,
    output_dir: Path,
    languages: list[str] | None = None,
    runs: list[str] | None = None,
) -> dict[tuple[str, str], int]:
    """
    Convert a participant zip submission into per-language, per-run JSONL files.

    Parameters
    ----------
    zip_path : Path
        Path to <team_name>.zip.
    reference_dir : Path
        Root of the multiclinsum2 test set directory.
    output_dir : Path
        Root output directory. JSONL files are written to
        <output_dir>/<team_name>/<language>/<run>/results.jsonl.
    languages : list[str], optional
        If given, only process these language codes. Otherwise all language
        folders found in the zip are processed.
    runs : list[str], optional
        If given, only process these run names (e.g. ["run1"]). Otherwise all
        run folders found in the zip are processed.

    Returns
    -------
    dict[tuple[str, str], int]
        Mapping of (language, run) → number of records written.

    Expected zip structure
    ----------------------
        <team_name>.zip
        └── <team_name>/
            └── <language>/
                └── <run>/
                    ├── multiclinsum2_test_1_<language>_sum.txt
                    └── ...
    """
    team_name = zip_path.stem
    logger.info("Processing submission: team=%s, zip=%s", team_name, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Discover (language, run) pairs present in the zip.
        # Expected path structure: {team_name}/{lang}/{run}/{file}_sum.txt
        # → parts[1] = lang, parts[2] = run
        lang_run_pairs: set[tuple[str, str]] = set()
        for name in all_names:
            parts = PurePosixPath(name).parts
            if parts[-1].endswith("_sum.txt") and len(parts) >= 4:
                # parts: (team_name, lang, run, filename)
                lang_run_pairs.add((parts[1], parts[2]))

        if not lang_run_pairs:
            raise ValueError(
                f"No '*_sum.txt' files found in the expected structure inside {zip_path}. "
                "Expected: <team_name>/<language>/<run>/<case>_sum.txt"
            )

        found_langs = {lang for lang, _ in lang_run_pairs}
        found_runs = {run for _, run in lang_run_pairs}
        logger.info("Languages found in zip: %s", sorted(found_langs))
        logger.info("Runs found in zip: %s", sorted(found_runs))

        # Filter by requested languages and runs
        target_langs = sorted(languages) if languages else sorted(found_langs)
        target_runs = sorted(runs) if runs else sorted(found_runs)

        unknown_langs = set(target_langs) - found_langs
        if unknown_langs:
            logger.warning("Requested languages not found in zip: %s", sorted(unknown_langs))
            target_langs = [l for l in target_langs if l in found_langs]

        unknown_runs = set(target_runs) - found_runs
        if unknown_runs:
            logger.warning("Requested runs not found in zip: %s", sorted(unknown_runs))
            target_runs = [r for r in target_runs if r in found_runs]

        counts: dict[tuple[str, str], int] = {}

        for lang in target_langs:
            if not (reference_dir / lang).is_dir():
                logger.warning(
                    "[%s] Reference directory not found: %s. Skipping language.",
                    lang,
                    reference_dir / lang,
                )
                continue

            for run in target_runs:
                if (lang, run) not in lang_run_pairs:
                    logger.warning(
                        "[%s/%s] Not found in zip — skipping.", lang, run
                    )
                    continue

                # Collect generated summaries for this (lang, run) from the zip
                generated: dict[str, str] = {}
                for name in all_names:
                    parts = PurePosixPath(name).parts
                    # Match: {team_name}/{lang}/{run}/{filename}_sum.txt
                    if (
                        len(parts) == 4
                        and parts[1] == lang
                        and parts[2] == run
                        and parts[-1].endswith("_sum.txt")
                    ):
                        filename = parts[-1]
                        try:
                            case_id = _case_id_from_sum_filename(filename)
                        except ValueError as exc:
                            logger.warning(
                                "Skipping malformed filename '%s': %s", filename, exc
                            )
                            continue
                        generated[case_id] = zf.read(name).decode("utf-8").strip()

                if not generated:
                    logger.warning("[%s/%s] No summary files collected. Skipping.", lang, run)
                    continue

                logger.info("[%s/%s] Found %d summary files.", lang, run, len(generated))

                records, warnings = convert_language(lang, generated, reference_dir)
                for w in warnings:
                    logger.warning(w)

                if not records:
                    logger.warning("[%s/%s] No records produced. Skipping output.", lang, run)
                    continue

                # Write JSONL
                out_path = output_dir / team_name / lang / run / "results.jsonl"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as fh:
                    for record in records:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                counts[(lang, run)] = len(records)
                logger.info("[%s/%s] Wrote %d records → %s", lang, run, len(records), out_path)

                unmatched = set(generated) - {r["case_id"] for r in records}
                if unmatched:
                    logger.warning(
                        "[%s/%s] %d participant files had no matching reference: %s",
                        lang,
                        run,
                        len(unmatched),
                        sorted(unmatched)[:10],
                    )

    logger.info(
        "Done. Records written per (language, run): %s",
        {f"{l}/{r}": v for (l, r), v in sorted(counts.items())},
    )
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a participant .zip submission to evaluation-ready JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to the participant zip file (<team_name>.zip).",
    )
    parser.add_argument(
        "--reference_dir",
        required=True,
        help="Root of the multiclinsum2 test set (contains one folder per language).",
    )
    parser.add_argument(
        "--output_dir",
        default="inputs",
        help="Root output directory. JSONL files are written to "
             "<output_dir>/<team_name>/<language>/results.jsonl.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Language codes to process (default: all languages found in the zip).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Run names to process, e.g. run1 run2 (default: all runs found in the zip).",
    )
    return parser.parse_args()


def main() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    args = _parse_args()
    prepare_submission(
        zip_path=Path(args.submission),
        reference_dir=Path(args.reference_dir),
        output_dir=Path(args.output_dir),
        languages=args.languages,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
