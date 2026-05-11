from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer

import config
from metrics.classic.base import BaseMetric

logger = logging.getLogger(__name__)


class BARTScorer(BaseMetric):
    """
    BARTScore: mean token log-likelihood using BART-large-cnn.

    documents: source texts (full clinical case reports)
    summaries: system-generated summaries

    Returns two directions:
      bartscore_src_hypo  P(summary | source)  — faithfulness proxy
      bartscore_hypo_src  P(source | summary)  — recall proxy

    Scores are negative floats (mean log-likelihood per token, ~-1 to -4).
    Higher (less negative) = better.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        self.device = device or config.BARTSCORE_DEVICE
        self.batch_size = batch_size or config.BARTSCORE_BATCH_SIZE
        self.max_length = max_length or config.BARTSCORE_MAX_LENGTH
        model_path = model_name or config.BARTSCORE_MODEL

        logger.info("Loading BARTScorer from %s on %s ...", model_path, self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("BARTScorer ready.")

    def _log_likelihood(self, sources: list[str], targets: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(sources), self.batch_size):
            src_batch = sources[i : i + self.batch_size]
            tgt_batch = targets[i : i + self.batch_size]

            src_enc = self.tokenizer(
                src_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            tgt_enc = self.tokenizer(
                tgt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            tgt_ids = tgt_enc["input_ids"]
            labels = tgt_ids.masked_fill(tgt_ids == self.tokenizer.pad_token_id, -100)

            with torch.no_grad():
                output = self.model(
                    input_ids=src_enc["input_ids"],
                    attention_mask=src_enc["attention_mask"],
                    labels=labels,
                )
                logits = output.logits  # (B, T, V)

            log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

            for b in range(len(src_batch)):
                valid_mask = labels[b] != -100
                token_ids = tgt_ids[b][valid_mask]
                token_log_probs = log_probs[b][valid_mask]
                selected = token_log_probs[range(len(token_ids)), token_ids]
                scores.append(round(selected.mean().item(), 6))

        return scores

    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        logger.info("BARTScore src→hypo (%d pairs)...", len(documents))
        src_hypo = self._log_likelihood(documents, summaries)
        logger.info("BARTScore hypo→src (%d pairs)...", len(documents))
        hypo_src = self._log_likelihood(summaries, documents)
        return {"bartscore_src_hypo": src_hypo, "bartscore_hypo_src": hypo_src}
