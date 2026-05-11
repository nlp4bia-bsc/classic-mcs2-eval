from __future__ import annotations

import logging

import config
from metrics.classic.base import BaseMetric

logger = logging.getLogger(__name__)


class SummaCZSMetric(BaseMetric):
    """
    SummaC ZS: zero-shot NLI-based factual consistency.

    documents: source texts (full clinical case reports)
    summaries: system-generated summaries

    Scores in [0, 1]. Higher = more factually consistent with the source.

    Note: requires `summac` package. Install with --no-deps to avoid
    the torch==2.1.1 hard-pin conflict with vLLM's torch version.
    """

    def __init__(
        self,
        model_name: str = "vitc",
        granularity: str | None = None,
        device: str | None = None,
    ):
        from summac.model_summac import SummaCZS  # deferred — only required at use time

        self._model = SummaCZS(
            model_name=model_name,
            granularity=granularity or config.SUMMAC_GRANULARITY,
            device=device or config.SUMMAC_DEVICE,
        )
        logger.info("SummaCZS ready (model=%s, granularity=%s, device=%s).", model_name, granularity or config.SUMMAC_GRANULARITY, device or config.SUMMAC_DEVICE)

    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        result = self._model.score(documents, summaries)
        return {"summac_zs": [round(float(v), 6) for v in result["scores"]]}


class SummaCConvMetric(BaseMetric):
    """
    SummaC Conv: conv-model NLI-based factual consistency.

    documents: source texts (full clinical case reports)
    summaries: system-generated summaries

    Scores in [0, 1]. Requires pre-trained conv weights at
    config.SUMMAC_CONV_MODEL_PATH (summac_conv_vitc_sent_perc_e.bin).
    If path is empty the summac library falls back to its default download path.
    """

    def __init__(
        self,
        device: str | None = None,
        start_file: str | None = None,
    ):
        from summac.model_summac import SummaCConv  # deferred — only required at use time

        start = start_file or (config.SUMMAC_CONV_MODEL_PATH or None)
        self._model = SummaCConv(
            models=["vitc"],
            granularity=config.SUMMAC_GRANULARITY,
            nli_labels="e",
            device=device or config.SUMMAC_DEVICE,
            start_file=start,
            agg="mean",
        )
        logger.info("SummaCConv ready (device=%s, start_file=%s).", device or config.SUMMAC_DEVICE, start)

    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        result = self._model.score(documents, summaries)
        return {"summac_conv": [round(float(v), 6) for v in result["scores"]]}
