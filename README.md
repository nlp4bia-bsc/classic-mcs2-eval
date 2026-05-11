# multiclinsum2 — Classic Metrics Evaluation

Evaluation framework for the multiclinsum2 clinical summarisation shared task.
Scores participant-generated summaries against gold references using classic NLP metrics.

**Metrics:** ROUGE-1, ROUGE-2, ROUGE-Lsum, BERTScore (always); BARTScore, SummaC (optional, flag-gated).

All metrics are reported independently — no combined score.

---

## Repository layout

```
classic_summ_evaluation/
├── config.py                        # All configuration constants
├── schemas.py                       # Pydantic v2 data models
├── runner.py                        # Evaluation pipeline entry point
├── prepare_submission.py            # Converts participant zip to JSONL
├── pyproject.toml                   # Dependencies (uv)
└── metrics/
    └── classic/
        ├── base.py                  # BaseMetric ABC
        ├── rouge_metric.py          # ROUGE-1, ROUGE-2, ROUGE-Lsum
        ├── bertscore_metric.py      # BERTScore (multilingual mDeBERTa)
        ├── bartscore_metric.py      # BARTScore (optional, --use_bartscore)
        └── summac_metric.py         # SummaC ZS/Conv (optional, --use_summac)
```

---

## Setup

```bash
uv sync                        # core deps
uv sync --extra analysis       # + notebooks/plotting
```

---

## End-to-end workflow

```
participant zip
      │
      ▼
prepare_submission.py   →   inputs/<team>/<lang>/<run>/results.jsonl
      │
      ▼
runner.py               →   outputs/<team>/<lang>/<run>/{classic_metrics.jsonl, report.json}
```

---

## Step 1 — Convert participant submission

Participants submit a zip with this structure:

```
{team_name}.zip
└── {team_name}/
    └── {language}/
        └── run1/
            ├── multiclinsum2_test_1_{language}_sum.txt
            ├── multiclinsum2_test_2_{language}_sum.txt
            └── ...
```

Convert to evaluation-ready JSONL:

```bash
python prepare_submission.py \
    --submission submissions/team_x.zip \
    --reference_dir ../data/multiclinsum2_test_set \
    --output_dir inputs/
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--submission` | *(required)* | Path to `<team_name>.zip` |
| `--reference_dir` | *(required)* | Root of the test set (one folder per language with full cases and gold summaries) |
| `--output_dir` | `inputs/` | Root output directory |
| `--languages` | all found | Process only specific language codes |
| `--runs` | all found | Process only specific runs (e.g. `run1`) |

**Output:**

```
inputs/
└── team_x/
    ├── en/
    │   ├── run1/results.jsonl
    │   └── run2/results.jsonl
    └── es/
        └── run1/results.jsonl
```

---

## Step 2 — Run evaluation

### All languages for one team (recommended)

```bash
python runner.py --team_dir inputs/team_x
```

Automatically discovers both path structures:
- `<lang>/results.jsonl` — baseline outputs (no run subfolder)
- `<lang>/<run>/results.jsonl` — participant submissions

### Single language / single run

```bash
python runner.py --input inputs/team_x/en/run1/results.jsonl
```

### With optional neural metrics

```bash
python runner.py --input inputs/team_x/en/run1/results.jsonl \
    --use_bartscore \
    --use_summac
```

### Override metadata when path is non-standard

```bash
python runner.py \
    --input /path/to/results.jsonl \
    --team_name team_x \
    --language en \
    --output_dir outputs/
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Path to a single `results.jsonl` (mutually exclusive with `--team_dir`) |
| `--team_dir` | — | Team directory; evaluates all languages/runs found inside |
| `--output_dir` | from config | Root output directory |
| `--team_name` | inferred | Override team name |
| `--language` | inferred | Override language code (single `--input` only) |
| `--use_bartscore` | `False` | Compute BARTScore (BART-large-cnn, ~1.6 GB) |
| `--use_summac` | `False` | Compute SummaC ZS/Conv factual consistency |

---

## Input format

Each `results.jsonl` has one JSON object per line:

```json
{
  "case_id": "multiclinsum2_test_1_ca",
  "language": "ca",
  "full_case": "...",
  "reference_summary": "...",
  "generated_summary": "..."
}
```

Lines with `generated_summary` starting with `ERROR:` are skipped and logged as warnings.

---

## Output format

```
outputs/
└── {team_name}/
    └── {language}/
        └── {run}/            ← omitted for baseline outputs
            ├── classic_metrics.jsonl   # written first — survives a timeout
            └── report.json             # final aggregated report
```

`classic_metrics.jsonl` is written before `report.json` so a SLURM timeout does not discard computed scores.

Each `report.json` contains:
- `per_sample` — `SampleReport` for every evaluated case
- `classic_aggregated` — mean/std/min/max for each metric

---

## Data models (`schemas.py`)

| Model | Description |
|-------|-------------|
| `EvaluationInput` | One sample: source text, generated summary, reference summary, team and language |
| `ClassicMetricsOutput` | ROUGE-1/2/Lsum, BERTScore, and optional BARTScore/SummaC for one sample |
| `SampleReport` | Full evaluation of one sample |
| `MetricStats` | Descriptive statistics (mean, std, min, max) for one metric across all samples |
| `SystemReport` | Aggregated report for one team × language × run |

---

## Classic metrics

### ROUGE

ROUGE-1, ROUGE-2, and **ROUGE-Lsum** via `rouge_score`. ROUGE-Lsum splits on newlines before computing LCS — more appropriate than ROUGE-L for multi-sentence clinical summaries.

### BERTScore

`microsoft/mdeberta-v3-base` — single multilingual model covering all 15 evaluation languages. Scores are clamped to [0, 1].

Set `BERTSCORE_RESCALE_BASELINE = True` in `config.py` to rescale against a language-specific baseline for better cross-language comparability.

### BARTScore (optional)

Mean token log-likelihood using BART-large-cnn. Computes two directions:
- `bartscore_src_hypo` — P(summary | source): faithfulness proxy
- `bartscore_hypo_src` — P(source | summary): recall proxy

Scores are negative floats (~-1 to -4); higher (less negative) = better.

### SummaC (optional)

Zero-shot NLI-based factual consistency score against the source text. Returns:
- `summac_zs` — SummaC ZS (always computed when `--use_summac`)
- `summac_conv` — SummaC Conv (only if `SUMMAC_CONV_MODEL_PATH` is set in `config.py`)

---

## Configuration (`config.py`)

Critical settings to update before running:

| Setting | Description |
|---------|-------------|
| `BERTSCORE_MODEL` | Absolute path or HF Hub name for mDeBERTa-v3-base |
| `BERTSCORE_NUM_LAYERS` | Transformer layers for BERTScore (12 for mDeBERTa) |
| `OUTPUTS_DIR` | Root directory for output reports |
| `BARTSCORE_MODEL` | HF Hub name or local path for BART-large-cnn (`--use_bartscore`) |
| `BARTSCORE_DEVICE` | `"cuda"` or `"cpu"` |
| `SUMMAC_CONV_MODEL_PATH` | Path to `summac_conv_vitc_sent_perc_e.bin`; empty = skip Conv |
| `SUMMAC_DEVICE` | `"cuda"` or `"cpu"` |

---

## Testing

```bash
uv run pytest tests/ -v
```

No GPU or network access required. `BERTScoreMetric` is mocked; only `rouge_score` runs for real.

| File | What it covers |
|------|---------------|
| `tests/test_runner.py` | `_infer_metadata`, `load_inputs`, `_metric_stats`, `_build_system_report`, schema validation, `run()` integration |
| `tests/test_metrics.py` | `ROUGEMetric.score` — identical/empty/partial/mismatch/stemmer/custom types |

---

## Adding a new classic metric

1. Create `metrics/classic/my_metric.py` inheriting from `BaseMetric`
2. Implement `score(documents, summaries) -> dict[str, list[float]]`
3. Add the metric key(s) to `ClassicMetricsOutput` in `schemas.py`
4. Call `all_scores.update(MyMetric().score(...))` in `runner.py`
5. Add tests in `tests/test_metrics.py`
