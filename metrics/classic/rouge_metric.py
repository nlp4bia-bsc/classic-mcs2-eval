from __future__ import annotations

from rouge_score import rouge_scorer

from metrics.classic.base import BaseMetric

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeLsum"]


class ROUGEMetric(BaseMetric):
    """
    ROUGE-1, ROUGE-2, ROUGE-Lsum F1 scores.

    documents: gold reference summaries
    summaries: system-generated summaries

    ROUGE-Lsum splits on newlines before LCS — more appropriate than ROUGE-L
    for multi-sentence clinical summaries.
    """

    def __init__(self, rouge_types: list[str] | None = None, use_stemmer: bool = False):
        self.rouge_types = rouge_types or _ROUGE_TYPES
        self._scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)

    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        results: dict[str, list[float]] = {rt: [] for rt in self.rouge_types}
        for ref, hyp in zip(documents, summaries):
            pair = self._scorer.score(target=ref, prediction=hyp)
            for rt in self.rouge_types:
                results[rt].append(round(pair[rt].fmeasure, 6))
        return results
