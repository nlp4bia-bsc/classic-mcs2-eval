"""
schemas.py — Pydantic v2 data models for the multiclinsum2 evaluation framework.

Data flow:
    EvaluationInput
        └── → ClassicMetricsOutput   (via classic metrics pipeline)
                ↓
            SampleReport            (one per input sample)
                ↓
            SystemReport            (one per team × language)

All metrics are reported as independent scores — there is no combined final score.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    """
    Input to any metric.

    Represents a single prediction from one participant team,
    paired with the gold reference summary and the original source text.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sample_id": "42",
                    "team_name": "bsc-nlp",
                    "language": "es",
                    "source_text": "Paciente varón de 65 años ingresa por disnea...",
                    "generated_summary": "Hombre de 65 años con disnea aguda...",
                    "reference_summary": "Varón de 65 años con insuficiencia cardíaca...",
                }
            ]
        }
    }

    sample_id: str = Field(..., description="Unique identifier of the clinical case sample.")
    team_name: str = Field(..., description="Participant team identifier.")
    language: str = Field(..., description="ISO 639-1 language code of the text (e.g. 'es', 'fr', 'ca').")
    source_text: str = Field(..., description="Original medical case report (full text).")
    generated_summary: str = Field(..., description="System-generated summary from the participant.")
    reference_summary: str = Field(..., description="Gold standard reference summary.")


class ClassicMetricsOutput(BaseModel):
    """
    ROUGE and BERTScore results for a single sample.

    All scores are in [0, 1]. BERTScore is rescaled from its raw range
    to [0, 1] before storage.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sample_id": "42",
                    "team_name": "bsc-nlp",
                    "language": "es",
                    "rouge1": 0.61,
                    "rouge2": 0.38,
                    "rougeLsum": 0.55,
                    "bertscore": 0.82,
                }
            ]
        }
    }

    sample_id: str = Field(..., description="Unique identifier of the evaluated sample.")
    team_name: str = Field(..., description="Participant team identifier.")
    language: str = Field(..., description="ISO 639-1 language code.")
    rouge1: float = Field(..., ge=0.0, le=1.0, description="ROUGE-1 F1 score.")
    rouge2: float = Field(..., ge=0.0, le=1.0, description="ROUGE-2 F1 score.")
    rougeLsum: float = Field(..., ge=0.0, le=1.0, description="ROUGE-Lsum F1 score.")
    bertscore: float = Field(
        ..., ge=0.0, le=1.0,
        description="BERTScore F1 rescaled to [0, 1].",
    )
    bartscore_src_hypo: float | None = Field(
        default=None,
        description=(
            "BARTScore mean token log-likelihood P(summary|source). "
            "Negative float (~-1 to -4); higher (less negative) = better faithfulness proxy."
        ),
    )
    bartscore_hypo_src: float | None = Field(
        default=None,
        description=(
            "BARTScore mean token log-likelihood P(source|summary). "
            "Negative float (~-1 to -4); higher (less negative) = better recall proxy."
        ),
    )
    summac_zs: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="SummaC ZS NLI-based factual consistency score vs. source text.",
    )
    summac_conv: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="SummaC Conv NLI-based factual consistency score vs. source text.",
    )


class SampleReport(BaseModel):
    """Full evaluation of one sample: classic metrics only."""

    sample_id: str = Field(..., description="Unique identifier of the evaluated sample.")
    team_name: str = Field(..., description="Participant team identifier.")
    language: str = Field(..., description="ISO 639-1 language code.")
    classic_metrics: ClassicMetricsOutput = Field(
        ..., description="ROUGE and BERTScore results for this sample.",
    )


class MetricStats(BaseModel):
    """Descriptive statistics for a single metric across all evaluated samples."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"mean": 0.62, "std": 0.08, "min": 0.41, "max": 0.79}
            ]
        }
    }

    mean: float = Field(..., description="Mean score across all samples.")
    std: float = Field(..., description="Standard deviation.")
    min: float = Field(..., description="Minimum score observed.")
    max: float = Field(..., description="Maximum score observed.")


class SystemReport(BaseModel):
    """
    Full aggregated report for one team in one language.

    Summarises per-sample results and computes aggregate statistics
    independently for each classic metric. There is no combined final score.
    """

    team_name: str = Field(..., description="Participant team identifier.")
    language: str = Field(..., description="ISO 639-1 language code.")
    n_samples: int = Field(..., ge=0, description="Number of samples evaluated.")
    per_sample: list[SampleReport] = Field(
        ..., description="Full SampleReport for every evaluated sample.",
    )
    classic_aggregated: dict[str, MetricStats] = Field(
        ...,
        description=(
            "Aggregate statistics per classic metric across all samples. "
            "Keys: 'rouge1', 'rouge2', 'rougeLsum', 'bertscore' (always present); "
            "'bartscore_src_hypo', 'bartscore_hypo_src', 'summac_zs', 'summac_conv' (optional)."
        ),
    )
