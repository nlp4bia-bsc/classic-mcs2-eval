"""Tests for metrics/classic/ — ROUGE only (BERTScore requires GPU/model)."""

from __future__ import annotations

import pytest

from metrics.classic.rouge_metric import ROUGEMetric


# ---------------------------------------------------------------------------
# ROUGEMetric
# ---------------------------------------------------------------------------


class TestROUGEMetric:
    def setup_method(self):
        self.metric = ROUGEMetric()

    def test_returns_all_rouge_keys(self):
        scores = self.metric.score(["hello world"], ["hello world"])
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeLsum"}

    def test_identical_ref_and_hyp_scores_one(self):
        ref = "The patient presented with acute chest pain and was treated with aspirin."
        scores = self.metric.score([ref], [ref])
        assert scores["rouge1"][0] == pytest.approx(1.0)
        assert scores["rouge2"][0] == pytest.approx(1.0)
        assert scores["rougeLsum"][0] == pytest.approx(1.0)

    def test_empty_strings_score_zero(self):
        scores = self.metric.score([""], [""])
        assert scores["rouge1"][0] == pytest.approx(0.0)
        assert scores["rouge2"][0] == pytest.approx(0.0)
        assert scores["rougeLsum"][0] == pytest.approx(0.0)

    def test_empty_hypothesis_scores_zero(self):
        scores = self.metric.score(["The patient had fever."], [""])
        assert scores["rouge1"][0] == pytest.approx(0.0)

    def test_empty_reference_scores_zero(self):
        scores = self.metric.score([""], ["The patient had fever."])
        assert scores["rouge1"][0] == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        ref = "The patient presented with fever and headache."
        hyp = "The patient had fever."
        scores = self.metric.score([ref], [hyp])
        assert 0.0 < scores["rouge1"][0] < 1.0

    def test_list_length_matches_input(self):
        refs = ["ref one", "ref two", "ref three"]
        hyps = ["hyp one", "hyp two", "hyp three"]
        scores = self.metric.score(refs, hyps)
        for key in ("rouge1", "rouge2", "rougeLsum"):
            assert len(scores[key]) == 3

    def test_scores_rounded_to_six_decimal_places(self):
        scores = self.metric.score(["a b c d e"], ["a b c"])
        for key in ("rouge1", "rouge2", "rougeLsum"):
            val = scores[key][0]
            assert val == round(val, 6)

    def test_length_mismatch_truncates_to_shorter(self):
        refs = ["ref one", "ref two", "ref three"]
        hyps = ["hyp one", "hyp two"]
        scores = self.metric.score(refs, hyps)
        for key in ("rouge1", "rouge2", "rougeLsum"):
            assert len(scores[key]) == 2

    def test_scores_in_zero_one_range(self):
        pairs = [
            ("acute myocardial infarction treated with PCI", "MI treated with stenting"),
            ("fever headache neck stiffness", "meningitis"),
            ("dilated cardiomyopathy ejection fraction", "reduced EF cardiomyopathy"),
        ]
        refs, hyps = zip(*pairs)
        scores = self.metric.score(list(refs), list(hyps))
        for key in ("rouge1", "rouge2", "rougeLsum"):
            for v in scores[key]:
                assert 0.0 <= v <= 1.0

    def test_use_stemmer_option(self):
        metric = ROUGEMetric(use_stemmer=True)
        scores = metric.score(["running patients treated"], ["run patient treat"])
        assert scores["rouge1"][0] > 0.0

    def test_custom_rouge_types(self):
        metric = ROUGEMetric(rouge_types=["rouge1"])
        scores = metric.score(["hello"], ["hello"])
        assert set(scores.keys()) == {"rouge1"}
        assert "rouge2" not in scores
