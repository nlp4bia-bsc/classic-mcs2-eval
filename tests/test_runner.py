"""Tests for runner.py — metadata inference, input loading, aggregation, and integration."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from runner import (
    _build_system_report,
    _infer_metadata,
    _metric_stats,
    load_inputs,
    run,
)
from schemas import (
    ClassicMetricsOutput,
    MetricStats,
    SampleReport,
    SystemReport,
)


# ---------------------------------------------------------------------------
# _infer_metadata
# ---------------------------------------------------------------------------


class TestInferMetadata:
    def test_with_run(self):
        p = Path("/root/team_x/en/run1/results.jsonl")
        team, lang, run_name = _infer_metadata(p)
        assert team == "team_x"
        assert lang == "en"
        assert run_name == "run1"

    def test_without_run(self):
        p = Path("/root/team_x/en/results.jsonl")
        team, lang, run_name = _infer_metadata(p)
        assert team == "team_x"
        assert lang == "en"
        assert run_name is None

    def test_short_path_fallback(self):
        p = Path("results.jsonl")
        team, lang, run_name = _infer_metadata(p)
        assert team == "unknown_team"
        assert lang == "unknown"
        assert run_name is None

    def test_two_part_path_fallback(self):
        p = Path("en/results.jsonl")
        team, lang, run_name = _infer_metadata(p)
        assert team == "unknown_team"
        assert lang == "unknown"
        assert run_name is None

    def test_run_name_not_confused_with_language(self):
        # "run2" is not a language code, so it should be treated as a run dir
        p = Path("/data/teamA/fr/run2/results.jsonl")
        team, lang, run_name = _infer_metadata(p)
        assert team == "teamA"
        assert lang == "fr"
        assert run_name == "run2"

    def test_all_supported_languages_detected_as_no_run(self):
        for lang_code in ("es", "ca", "de", "pt", "it"):
            p = Path(f"/root/myteam/{lang_code}/results.jsonl")
            team, lang, run_name = _infer_metadata(p)
            assert lang == lang_code
            assert run_name is None


# ---------------------------------------------------------------------------
# load_inputs
# ---------------------------------------------------------------------------


VALID_RECORD = {
    "case_id": "t001",
    "language": "en",
    "full_case": "Patient presented with fever.",
    "reference_summary": "Fever case.",
    "generated_summary": "Patient had fever.",
}


def _write_jsonl(path: Path, records: list[dict | str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write((rec if isinstance(rec, str) else json.dumps(rec)) + "\n")


class TestLoadInputs:
    def test_valid_jsonl(self, tmp_path):
        p = tmp_path / "results.jsonl"
        _write_jsonl(p, [VALID_RECORD] * 3)
        inputs = load_inputs(p, team_name="team", language="en")
        assert len(inputs) == 3
        assert inputs[0].sample_id == "t001"
        assert inputs[0].team_name == "team"
        assert inputs[0].language == "en"

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "results.jsonl"
        bad = {k: v for k, v in VALID_RECORD.items() if k != "generated_summary"}
        _write_jsonl(p, [bad])
        with pytest.raises(ValueError, match="missing fields"):
            load_inputs(p, team_name="team", language="en")

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "results.jsonl"
        p.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON parse error"):
            load_inputs(p, team_name="team", language="en")

    def test_error_prefix_lines_skipped(self, tmp_path):
        p = tmp_path / "results.jsonl"
        error_record = {**VALID_RECORD, "generated_summary": "ERROR: model crashed"}
        _write_jsonl(p, [VALID_RECORD, error_record, VALID_RECORD])
        inputs = load_inputs(p, team_name="team", language="en")
        assert len(inputs) == 2

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "results.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(json.dumps(VALID_RECORD) + "\n")
            fh.write("\n")
        inputs = load_inputs(p, team_name="team", language="en")
        assert len(inputs) == 1

    def test_source_and_reference_populated(self, tmp_path):
        p = tmp_path / "results.jsonl"
        _write_jsonl(p, [VALID_RECORD])
        inp = load_inputs(p, team_name="team", language="en")[0]
        assert inp.source_text == VALID_RECORD["full_case"]
        assert inp.reference_summary == VALID_RECORD["reference_summary"]
        assert inp.generated_summary == VALID_RECORD["generated_summary"]


# ---------------------------------------------------------------------------
# _metric_stats
# ---------------------------------------------------------------------------


class TestMetricStats:
    def test_empty_list(self):
        stats = _metric_stats([])
        assert stats == MetricStats(mean=0.0, std=0.0, min=0.0, max=0.0)

    def test_single_value(self):
        stats = _metric_stats([0.75])
        assert stats.mean == 0.75
        assert stats.std == 0.0
        assert stats.min == 0.75
        assert stats.max == 0.75

    def test_multiple_values_mean_and_extremes(self):
        vals = [0.4, 0.6, 0.8]
        stats = _metric_stats(vals)
        assert stats.mean == round(statistics.mean(vals), 6)
        assert stats.min == round(min(vals), 6)
        assert stats.max == round(max(vals), 6)

    def test_multiple_values_std(self):
        vals = [0.4, 0.6, 0.8]
        stats = _metric_stats(vals)
        assert stats.std == round(statistics.pstdev(vals), 6)

    def test_rounding_to_six_decimal_places(self):
        vals = [1 / 3, 2 / 3]
        stats = _metric_stats(vals)
        assert stats.mean == round(statistics.mean(vals), 6)
        assert len(str(stats.mean).split(".")[-1]) <= 6


# ---------------------------------------------------------------------------
# _build_system_report
# ---------------------------------------------------------------------------


def _make_sample_report(sample_id: str, rouge1: float = 0.5, **optional) -> SampleReport:
    cm = ClassicMetricsOutput(
        sample_id=sample_id,
        team_name="team",
        language="en",
        rouge1=rouge1,
        rouge2=0.3,
        rougeLsum=0.4,
        bertscore=0.7,
        **optional,
    )
    return SampleReport(
        sample_id=sample_id,
        team_name="team",
        language="en",
        classic_metrics=cm,
    )


class TestBuildSystemReport:
    def test_n_samples(self):
        reports = [_make_sample_report(f"s{i}") for i in range(4)]
        system = _build_system_report("team", "en", reports)
        assert system.n_samples == 4

    def test_always_present_keys(self):
        reports = [_make_sample_report("s0")]
        system = _build_system_report("team", "en", reports)
        for key in ("rouge1", "rouge2", "rougeLsum", "bertscore"):
            assert key in system.classic_aggregated

    def test_optional_key_absent_when_all_none(self):
        reports = [_make_sample_report("s0")]
        system = _build_system_report("team", "en", reports)
        assert "summac_zs" not in system.classic_aggregated
        assert "bartscore_src_hypo" not in system.classic_aggregated

    def test_optional_key_present_when_any_not_none(self):
        reports = [_make_sample_report("s0", summac_zs=0.85)]
        system = _build_system_report("team", "en", reports)
        assert "summac_zs" in system.classic_aggregated
        assert system.classic_aggregated["summac_zs"].mean == pytest.approx(0.85)

    def test_aggregation_correctness(self):
        reports = [
            _make_sample_report("s0", rouge1=0.4),
            _make_sample_report("s1", rouge1=0.6),
            _make_sample_report("s2", rouge1=0.8),
        ]
        system = _build_system_report("team", "en", reports)
        stats = system.classic_aggregated["rouge1"]
        assert stats.mean == pytest.approx(round(statistics.mean([0.4, 0.6, 0.8]), 6))
        assert stats.min == pytest.approx(0.4, abs=1e-6)
        assert stats.max == pytest.approx(0.8, abs=1e-6)

    def test_team_and_language_forwarded(self):
        system = _build_system_report("alpha_team", "fr", [_make_sample_report("s0")])
        assert system.team_name == "alpha_team"
        assert system.language == "fr"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_rouge1_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ClassicMetricsOutput(
                sample_id="x", team_name="t", language="en",
                rouge1=1.5, rouge2=0.3, rougeLsum=0.4, bertscore=0.7,
            )

    def test_bertscore_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ClassicMetricsOutput(
                sample_id="x", team_name="t", language="en",
                rouge1=0.5, rouge2=0.3, rougeLsum=0.4, bertscore=1.1,
            )

    def test_bartscore_fields_accept_none(self):
        cm = ClassicMetricsOutput(
            sample_id="x", team_name="t", language="en",
            rouge1=0.5, rouge2=0.3, rougeLsum=0.4, bertscore=0.7,
            bartscore_src_hypo=None,
            bartscore_hypo_src=None,
        )
        assert cm.bartscore_src_hypo is None
        assert cm.bartscore_hypo_src is None

    def test_bartscore_fields_accept_negative_float(self):
        cm = ClassicMetricsOutput(
            sample_id="x", team_name="t", language="en",
            rouge1=0.5, rouge2=0.3, rougeLsum=0.4, bertscore=0.7,
            bartscore_src_hypo=-2.34,
            bartscore_hypo_src=-1.87,
        )
        assert cm.bartscore_src_hypo == pytest.approx(-2.34)

    def test_summac_fields_accept_none(self):
        cm = ClassicMetricsOutput(
            sample_id="x", team_name="t", language="en",
            rouge1=0.5, rouge2=0.3, rougeLsum=0.4, bertscore=0.7,
            summac_zs=None,
            summac_conv=None,
        )
        assert cm.summac_zs is None
        assert cm.summac_conv is None


# ---------------------------------------------------------------------------
# Integration: run()
# ---------------------------------------------------------------------------

SAMPLE_JSONL = (
    Path(__file__).parent.parent / "inputs" / "test_team" / "en" / "run1" / "results.jsonl"
)


@pytest.mark.skipif(not SAMPLE_JSONL.exists(), reason="sample JSONL not present")
class TestRunIntegration:
    def _mock_bertscore(self, n: int):
        mock = MagicMock()
        mock.return_value.score.return_value = {"bertscore": [0.75] * n}
        return mock

    def test_report_json_created(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            report = run(
                input_path=SAMPLE_JSONL,
                output_dir=tmp_path,
            )
        report_path = tmp_path / "test_team" / "en" / "run1" / "report.json"
        assert report_path.exists()

    def test_classic_metrics_jsonl_written_before_report(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        run_dir = tmp_path / "test_team" / "en" / "run1"
        classic_mtime = os.stat(run_dir / "classic_metrics.jsonl").st_mtime_ns
        report_mtime = os.stat(run_dir / "report.json").st_mtime_ns
        assert classic_mtime <= report_mtime

    def test_n_samples_equals_five(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            report = run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        assert report.n_samples == 5

    def test_all_classic_keys_present(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            report = run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        for key in ("rouge1", "rouge2", "rougeLsum", "bertscore"):
            assert key in report.classic_aggregated

    def test_bertscore_aggregated_from_mock(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            report = run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        assert report.classic_aggregated["bertscore"].mean == pytest.approx(0.75)

    def test_report_json_deserialises_to_system_report(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        report_path = tmp_path / "test_team" / "en" / "run1" / "report.json"
        loaded = SystemReport.model_validate_json(report_path.read_text())
        assert loaded.n_samples == 5

    def test_classic_metrics_jsonl_has_five_lines(self, tmp_path):
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            run(input_path=SAMPLE_JSONL, output_dir=tmp_path)
        classic_path = tmp_path / "test_team" / "en" / "run1" / "classic_metrics.jsonl"
        lines = [l for l in classic_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

    def test_output_dir_override(self, tmp_path):
        custom_out = tmp_path / "custom_outputs"
        with patch("runner.BERTScoreMetric", self._mock_bertscore(5)):
            run(input_path=SAMPLE_JSONL, output_dir=custom_out)
        assert (custom_out / "test_team" / "en" / "run1" / "report.json").exists()
