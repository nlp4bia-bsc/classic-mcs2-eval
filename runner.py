"""
runner.py — Single entry point for the classic metrics evaluation pipeline.

Orchestrates:
  1. Loading EvaluationInput objects from a JSONL file (baseline model outputs)
  2. Running classic metrics (ROUGE + BERTScore, optionally BARTScore/SummaC)
  3. Assembling SampleReport and SystemReport
  4. Saving the SystemReport as report.json under outputs/

Usage
-----
Single language:
    python runner.py --input inputs/team_x/en/run1/results.jsonl

With optional neural classic metrics:
    python runner.py --input inputs/team_x/en/run1/results.jsonl --use_bartscore --use_summac

All languages for a team:
    python runner.py --team_dir inputs/team_x

Override metadata when path structure is non-standard:
    python runner.py --input /path/to/results.jsonl --team_name team_x --language en --output_dir outputs/

Input JSONL fields (produced by the baseline model):
    case_id            — str, unique case identifier
    language           — str, ISO 639-1 language code
    full_case          — str, original clinical case report
    reference_summary  — str, gold reference summary
    generated_summary  — str, system-generated summary
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import config
from schemas import ClassicMetricsOutput, EvaluationInput, MetricStats, SampleReport, SystemReport
from metrics.classic.rouge_metric import ROUGEMetric
from metrics.classic.bertscore_metric import BERTScoreMetric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_inputs(jsonl_path: Path, team_name: str, language: str) -> list[EvaluationInput]:
    """
    Load evaluation inputs from a JSONL file produced by the baseline model.

    Parameters
    ----------
    jsonl_path : Path
        Path to a results.jsonl file.
    team_name : str
        Participant team identifier (used to populate EvaluationInput.team_name).
    language : str
        ISO 639-1 language code (used to populate EvaluationInput.language).

    Returns
    -------
    list[EvaluationInput]

    Raises
    ------
    ValueError
        If a line is missing required fields or cannot be parsed.
    """
    inputs: list[EvaluationInput] = []
    required_fields = {"case_id", "full_case", "reference_summary", "generated_summary"}

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error on line {line_no} of {jsonl_path}: {exc}"
                ) from exc

            missing = required_fields - record.keys()
            if missing:
                raise ValueError(
                    f"Line {line_no} of {jsonl_path} is missing fields: {missing}"
                )

            # Skip lines that were written as errors by the baseline model
            generated = record["generated_summary"]
            if isinstance(generated, str) and generated.startswith("ERROR:"):
                logger.warning(
                    "Skipping sample %s — baseline inference produced an error: %s",
                    record["case_id"],
                    generated[:120],
                )
                continue

            inputs.append(
                EvaluationInput(
                    sample_id=str(record["case_id"]),
                    team_name=team_name,
                    language=language,
                    source_text=record["full_case"],
                    generated_summary=generated,
                    reference_summary=record["reference_summary"],
                )
            )

    logger.info("Loaded %d evaluation inputs from %s", len(inputs), jsonl_path)
    return inputs


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_report(report: SystemReport, output_path: Path) -> None:
    """
    Serialise a SystemReport to a JSON file.

    Parent directories are created if they do not exist.

    Parameters
    ----------
    report : SystemReport
    output_path : Path
        Destination file path (e.g. outputs/llama3.1-70b/ca/report.json).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))
    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _infer_metadata(jsonl_path: Path) -> tuple[str, str, str | None]:
    """
    Infer team_name, language, and optional run from the JSONL file path.

    Two supported structures:
      - Without run: <root>/<team_name>/<language>/results.jsonl
      - With run:    <root>/<team_name>/<language>/<run>/results.jsonl

    Returns (team_name, language, run_name).
    run_name is None when there is no run subfolder.
    Falls back to ("unknown_team", "unknown", None) if path is too short.
    """
    parts = jsonl_path.parts
    # parts[-1] is "results.jsonl", so:
    #   no-run:  parts[-2]=lang, parts[-3]=team
    #   with-run: parts[-2]=run, parts[-3]=lang, parts[-4]=team
    if len(parts) >= 4 and parts[-2] not in ("en", "es", "fr", "pt", "it", "ru",
                                               "ca", "no", "da", "ro", "de", "el",
                                               "nl", "cs", "sv", "nb"):
        # parts[-2] looks like a run name (e.g. "run1"), not a language code
        run_name = parts[-2]
        language = parts[-3]
        team_name = parts[-4]
        return team_name, language, run_name
    if len(parts) >= 3:
        language = parts[-2]
        team_name = parts[-3]
        return team_name, language, None
    logger.warning(
        "Could not infer metadata from path '%s'. "
        "Use --team_name and --language to set them explicitly.",
        jsonl_path,
    )
    return "unknown_team", "unknown", None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _metric_stats(values: list[float]) -> MetricStats:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return MetricStats(mean=0.0, std=0.0, min=0.0, max=0.0)
    mean = statistics.mean(values)
    std = statistics.pstdev(values)  # population std for a fixed evaluation set
    return MetricStats(
        mean=round(mean, 6),
        std=round(std, 6),
        min=round(min(values), 6),
        max=round(max(values), 6),
    )


def _build_system_report(
    team_name: str,
    language: str,
    sample_reports: list[SampleReport],
) -> SystemReport:
    always_keys = ["rouge1", "rouge2", "rougeLsum", "bertscore"]
    optional_keys = ["bartscore_src_hypo", "bartscore_hypo_src", "summac_zs", "summac_conv"]

    classic_values: dict[str, list[float]] = {k: [] for k in always_keys}
    optional_values: dict[str, list[float]] = {k: [] for k in optional_keys}

    for report in sample_reports:
        cm = report.classic_metrics
        classic_values["rouge1"].append(cm.rouge1)
        classic_values["rouge2"].append(cm.rouge2)
        classic_values["rougeLsum"].append(cm.rougeLsum)
        classic_values["bertscore"].append(cm.bertscore)
        for k in optional_keys:
            v = getattr(cm, k, None)
            if v is not None:
                optional_values[k].append(v)

    classic_aggregated = {k: _metric_stats(v) for k, v in classic_values.items()}
    for k, vals in optional_values.items():
        if vals:
            classic_aggregated[k] = _metric_stats(vals)

    return SystemReport(
        team_name=team_name,
        language=language,
        n_samples=len(sample_reports),
        per_sample=sample_reports,
        classic_aggregated=classic_aggregated,
    )


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    team_name: str | None = None,
    language: str | None = None,
    output_dir: Path | None = None,
    use_bartscore: bool = False,
    use_summac: bool = False,
) -> SystemReport:
    """
    Full evaluation run for a single JSONL results file.

    Steps
    -----
    1. Infer or validate team_name, language, and run.
    2. Load EvaluationInput objects from the JSONL file.
    3. Compute ROUGE scores for all samples.
    4. Compute BERTScore in one batch for all samples.
    5. Optionally compute BARTScore and/or SummaC.
    6. Assemble SampleReport objects.
    7. Build a SystemReport with per-metric aggregate statistics.
    8. Save the SystemReport to:
         output_dir/<team_name>/<language>/report.json          (no run)
         output_dir/<team_name>/<language>/<run>/report.json    (with run)

    Parameters
    ----------
    input_path : Path
        Path to results.jsonl produced by the baseline model.
    team_name : str, optional
        Override inferred team name.
    language : str, optional
        Override inferred language code.
    output_dir : Path, optional
        Root output directory.  Defaults to config.OUTPUTS_DIR.

    Returns
    -------
    SystemReport
    """
    # --- Resolve team, language, and run ---
    inferred_team, inferred_lang, inferred_run = _infer_metadata(input_path)
    resolved_team = team_name or inferred_team
    resolved_lang = language or inferred_lang
    resolved_run = inferred_run  # None when no run subfolder

    if resolved_lang not in config.SUPPORTED_LANGUAGES:
        logger.warning(
            "Language '%s' is not in SUPPORTED_LANGUAGES. Proceeding anyway.",
            resolved_lang,
        )

    logger.info(
        "Starting evaluation — team=%s, language=%s, run=%s, input=%s",
        resolved_team,
        resolved_lang,
        resolved_run or "n/a",
        input_path,
    )

    # --- Resolve output directory early so intermediate files go to the right place ---
    out_dir = Path(output_dir) if output_dir else Path(config.OUTPUTS_DIR)
    run_dir = out_dir / resolved_team / resolved_lang
    if resolved_run:
        run_dir = run_dir / resolved_run
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    inputs = load_inputs(input_path, team_name=resolved_team, language=resolved_lang)
    if not inputs:
        logger.error("No valid inputs found in %s. Aborting.", input_path)
        sys.exit(1)

    # --- Classic metrics ---
    references = [inp.reference_summary for inp in inputs]
    sources = [inp.source_text for inp in inputs]
    generated = [inp.generated_summary for inp in inputs]

    logger.info("Computing ROUGE scores for %d samples...", len(inputs))
    t0 = time.perf_counter()
    all_scores: dict[str, list] = ROUGEMetric().score(references, generated)
    logger.info("ROUGE done in %.1f s", time.perf_counter() - t0)

    logger.info("Computing BERTScore (batch, model=%s)...", config.BERTSCORE_MODEL)
    t0 = time.perf_counter()
    all_scores.update(BERTScoreMetric(language=resolved_lang).score(references, generated))
    logger.info("BERTScore done in %.1f s", time.perf_counter() - t0)

    if use_bartscore:
        from metrics.classic.bartscore_metric import BARTScorer
        logger.info("Computing BARTScore (model=%s, device=%s)...", config.BARTSCORE_MODEL, config.BARTSCORE_DEVICE)
        t0 = time.perf_counter()
        try:
            bart = BARTScorer()
            all_scores.update(bart.score(sources, generated))
            del bart
            logger.info("BARTScore done in %.1f s", time.perf_counter() - t0)
        except Exception as exc:
            logger.error("BARTScore failed: %s — skipping.", exc)

    if use_summac:
        from metrics.classic.summac_metric import SummaCConvMetric, SummaCZSMetric
        logger.info("Computing SummaC ZS (device=%s)...", config.SUMMAC_DEVICE)
        t0 = time.perf_counter()
        try:
            all_scores.update(SummaCZSMetric().score(sources, generated))
            logger.info("SummaC ZS done in %.1f s", time.perf_counter() - t0)
        except Exception as exc:
            logger.error("SummaC ZS failed: %s — skipping.", exc)
        if config.SUMMAC_CONV_MODEL_PATH:
            logger.info("Computing SummaC Conv (model=%s)...", config.SUMMAC_CONV_MODEL_PATH)
            t0 = time.perf_counter()
            try:
                all_scores.update(SummaCConvMetric().score(sources, generated))
                logger.info("SummaC Conv done in %.1f s", time.perf_counter() - t0)
            except Exception as exc:
                logger.error("SummaC Conv failed: %s — skipping.", exc)

    classic_outputs = [
        ClassicMetricsOutput(
            sample_id=inputs[i].sample_id,
            team_name=inputs[i].team_name,
            language=inputs[i].language,
            rouge1=all_scores["rouge1"][i],
            rouge2=all_scores["rouge2"][i],
            rougeLsum=all_scores["rougeLsum"][i],
            bertscore=all_scores["bertscore"][i],
            bartscore_src_hypo=all_scores["bartscore_src_hypo"][i] if "bartscore_src_hypo" in all_scores else None,
            bartscore_hypo_src=all_scores["bartscore_hypo_src"][i] if "bartscore_hypo_src" in all_scores else None,
            summac_zs=all_scores["summac_zs"][i] if "summac_zs" in all_scores else None,
            summac_conv=all_scores["summac_conv"][i] if "summac_conv" in all_scores else None,
        )
        for i in range(len(inputs))
    ]

    # Save classic metrics checkpoint
    classic_path = run_dir / "classic_metrics.jsonl"
    with classic_path.open("w", encoding="utf-8") as fh:
        for co in classic_outputs:
            fh.write(co.model_dump_json() + "\n")
    logger.info("Classic metrics saved to %s", classic_path)

    # --- Assemble sample reports ---
    sample_reports = [
        SampleReport(
            sample_id=inp.sample_id,
            team_name=inp.team_name,
            language=inp.language,
            classic_metrics=classic,
        )
        for inp, classic in zip(inputs, classic_outputs)
    ]

    # --- Build and save system report ---
    system_report = _build_system_report(
        team_name=resolved_team,
        language=resolved_lang,
        sample_reports=sample_reports,
    )

    report_path = run_dir / "report.json"
    save_report(system_report, report_path)

    logger.info(
        "Evaluation complete — %d samples, report saved to %s",
        len(inputs),
        report_path,
    )
    return system_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multiclinsum2 classic metrics evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mutually exclusive: single file or whole team directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        default=None,
        help="Path to a single results.jsonl produced by the baseline model.",
    )
    input_group.add_argument(
        "--team_dir",
        default=None,
        help=(
            "Path to a team output directory whose subdirectories are language codes "
            "(e.g. outputs/medical-mt5-large/).  All <lang>/results.jsonl files found "
            "are evaluated in one job."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Root output directory (default: {config.OUTPUTS_DIR}).",
    )
    parser.add_argument(
        "--team_name",
        default=None,
        help="Team name override (default: inferred from input path).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "ISO 639-1 language code override (default: inferred from input path). "
            "Ignored when --team_dir is used."
        ),
    )
    parser.add_argument(
        "--use_bartscore",
        action="store_true",
        default=False,
        help="Compute BARTScore (BART-large-cnn). Loads ~1.6 GB model; uses config.BARTSCORE_DEVICE.",
    )
    parser.add_argument(
        "--use_summac",
        action="store_true",
        default=False,
        help="Compute SummaC ZS factual consistency score. Requires `summac` package.",
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
    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.input:
        run(
            input_path=Path(args.input),
            team_name=args.team_name,
            language=args.language,
            output_dir=out_dir,
            use_bartscore=args.use_bartscore,
            use_summac=args.use_summac,
        )

    else:
        team_dir = Path(args.team_dir)
        if not team_dir.is_dir():
            logger.error("--team_dir '%s' is not a directory.", team_dir)
            sys.exit(1)

        jsonl_files = sorted(team_dir.glob("*/results.jsonl")) + \
                      sorted(team_dir.glob("*/*/results.jsonl"))
        jsonl_files = sorted(set(jsonl_files))

        if not jsonl_files:
            logger.error(
                "No results.jsonl files found under '%s'. "
                "Expected: <team_dir>/<lang>/results.jsonl  or  "
                "<team_dir>/<lang>/<run>/results.jsonl",
                team_dir,
            )
            sys.exit(1)

        logger.info(
            "Found %d result file(s) under '%s':",
            len(jsonl_files),
            team_dir,
        )
        for p in jsonl_files:
            logger.info("  %s", p.relative_to(team_dir))

        failed: list[str] = []
        for jsonl_path in jsonl_files:
            lang = jsonl_path.parent.name
            logger.info("=" * 60)
            logger.info("Evaluating language: %s  (%s)", lang, jsonl_path)
            logger.info("=" * 60)
            try:
                run(
                    input_path=jsonl_path,
                    team_name=args.team_name,
                    language=None,
                    output_dir=out_dir,
                    use_bartscore=args.use_bartscore,
                    use_summac=args.use_summac,
                )
            except Exception as exc:
                logger.error("Evaluation FAILED for language '%s': %s", lang, exc, exc_info=True)
                failed.append(lang)

        if failed:
            logger.error("Evaluation failed for %d language(s): %s", len(failed), failed)
            sys.exit(1)
        logger.info("All %d language evaluations complete.", len(jsonl_files))


if __name__ == "__main__":
    main()
