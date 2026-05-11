# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv sync                        # core deps
uv sync --extra analysis       # + notebooks/plotting
```

## Running evaluation

Single language/run:
```bash
python runner.py --input inputs/team_x/en/run1/results.jsonl
```

With optional neural classic metrics:
```bash
python runner.py --input inputs/team_x/en/run1/results.jsonl --use_bartscore --use_summac
```

All languages for a team:
```bash
python runner.py --team_dir inputs/team_x
```

Override metadata when path structure is non-standard:
```bash
python runner.py --input /path/to/results.jsonl --team_name team_x --language en --output_dir outputs/
```

Convert participant zip to JSONL before evaluation:
```bash
python prepare_submission.py \
    --submission submissions/team_x.zip \
    --reference_dir ../data/multiclinsum2_test_set \
    --output_dir inputs/
```

## Running tests

```bash
uv run pytest tests/ -v          # all tests
uv run pytest tests/test_metrics.py   # ROUGE only (no model)
uv run pytest tests/test_runner.py    # runner unit + integration
```

No GPU or network access required. `BERTScoreMetric` is mocked in integration tests.

### Test layout

```
tests/
├── conftest.py          # adds project root to sys.path
├── test_runner.py       # _infer_metadata, load_inputs, _metric_stats,
│                        # _build_system_report, schema validation, run() integration
└── test_metrics.py      # ROUGEMetric.score (identical, empty, partial, mismatch)
```

## Architecture

### Data flow

```
participant zip → prepare_submission.py → inputs/<team>/<lang>/<run>/results.jsonl
                                                    ↓
                                              runner.py
                                                    ↓
                                          classic metrics
                                  (ROUGE-1/2/Lsum + BERTScore,
                                   optionally BARTScore/SummaC)
                                                    ↓
                                         ClassicMetricsOutput
                                                    ↓
                          outputs/<team>/<lang>/<run>/{classic_metrics.jsonl, report.json}
```

### Key design decisions

**Classic metrics share a `BaseMetric` ABC.** All metrics in `metrics/classic/` implement `score(documents, summaries) -> dict[str, list[float]]`. ROUGE/BERTScore use reference summaries as `documents`; BARTScore/SummaC use source texts as `documents` (faithfulness direction). `runner.py` collects all score dicts into `all_scores` then constructs `ClassicMetricsOutput` once.

**BARTScore and SummaC are flag-gated at runtime** (`--use_bartscore` / `--use_summac`) but always installed as regular dependencies. `summac`'s `torch==2.1.1` pin is overridden by the `override-dependencies` block in `pyproject.toml`.

**Fault tolerance.** `classic_metrics.jsonl` is written before `report.json` so a SLURM timeout doesn't discard computed scores.

**No combined score.** All metrics (rouge1, rouge2, rougeLsum, bertscore, bartscore_src_hypo, bartscore_hypo_src, summac_zs, summac_conv) are independent columns in the leaderboard. BARTScore/SummaC fields are `None` unless their respective flags were passed.

**Path inference.** `_infer_metadata()` in `runner.py` detects whether the JSONL sits at `<team>/<lang>/results.jsonl` (baseline, no run) or `<team>/<lang>/<run>/results.jsonl` (participant submission) by checking if the second-to-last directory matches a known language code. The output path mirrors this structure.

### Schemas (`schemas.py`)

Single source of truth for all data contracts. Pydantic v2. Key chain:
`EvaluationInput` → `ClassicMetricsOutput` → `SampleReport` → `SystemReport`

### Configuration (`config.py`)

Critical settings to update before running:
- `BERTSCORE_MODEL` — absolute path or HF Hub name for mDeBERTa-v3-base
- `BARTSCORE_MODEL` — HF Hub name or local path for BART-large-cnn (used with `--use_bartscore`)
- `SUMMAC_CONV_MODEL_PATH` — path to `summac_conv_vitc_sent_perc_e.bin`; leave empty to skip Conv (used with `--use_summac`)
- `OUTPUTS_DIR` — root directory for output reports

### Adding a new classic metric

1. Create `metrics/classic/my_metric.py` inheriting from `BaseMetric`
2. Implement `score(documents, summaries) -> dict[str, list[float]]`
3. Add the metric key to `ClassicMetricsOutput` in `schemas.py`
4. Call `all_scores.update(MyMetric().score(...))` in `runner.py`
