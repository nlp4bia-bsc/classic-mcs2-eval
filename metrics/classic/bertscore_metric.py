from __future__ import annotations

import logging

import config
from metrics.classic.base import BaseMetric

logger = logging.getLogger(__name__)


class BERTScoreMetric(BaseMetric):
    """
    BERTScore F1 using multilingual mDeBERTa-v3-base.

    documents: gold reference summaries
    summaries: system-generated summaries

    Monkey-patches bert_score's sent_encode to avoid OverflowError on mDeBERTa
    whose model_max_length is set to int(1e30) by the tokenizer config.
    """

    def __init__(self, language: str = "en"):
        self.language = language

    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        import bert_score.utils as _bsu
        from bert_score import score as bert_score_fn  # type: ignore[import]

        def _safe_sent_encode(tokenizer, a: str) -> list[int]:
            return tokenizer.encode(
                a,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
            )

        _bsu.sent_encode = _safe_sent_encode

        chunk_size = config.BERTSCORE_BATCH_SIZE
        logger.info(
            "Computing BERTScore for %d samples (lang=%s, model=%s, num_layers=%d, rescale=%s, chunk_size=%d)",
            len(documents),
            self.language,
            config.BERTSCORE_MODEL,
            config.BERTSCORE_NUM_LAYERS,
            config.BERTSCORE_RESCALE_BASELINE,
            chunk_size,
        )

        all_f1: list[float] = []
        for i in range(0, len(documents), chunk_size):
            chunk_refs = documents[i : i + chunk_size]
            chunk_hyps = summaries[i : i + chunk_size]
            logger.info(
                "BERTScore chunk %d-%d / %d",
                i + 1,
                min(i + chunk_size, len(documents)),
                len(documents),
            )
            _p, _r, f1 = bert_score_fn(
                cands=chunk_hyps,
                refs=chunk_refs,
                model_type=config.BERTSCORE_MODEL,
                num_layers=config.BERTSCORE_NUM_LAYERS,
                lang=self.language,
                rescale_with_baseline=config.BERTSCORE_RESCALE_BASELINE,
                use_fast_tokenizer=False,
                verbose=False,
            )
            all_f1.extend([max(0.0, min(1.0, round(float(v), 6))) for v in f1])

        return {"bertscore": all_f1}
