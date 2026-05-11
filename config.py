"""
config.py — Central configuration for the multiclinsum2 evaluation framework.

All constants are defined at module level. Import them directly:

    from config import BERTSCORE_MODEL, SUPPORTED_LANGUAGES

Each metric (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore) is reported as an
independent score. There is no combined final score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# BERTScore config
# ---------------------------------------------------------------------------

BERTSCORE_MODEL: str = "microsoft/mdeberta-v3-base"
# Multilingual DeBERTa — supports all 15 evaluation languages out of the box.
# Set to "microsoft/mdeberta-v3-base" to download from HuggingFace Hub.
# HuggingFace Hub name OR absolute path to a local model snapshot.
# If using a local path, bert_score cannot look up num_layers automatically —
# set BERTSCORE_NUM_LAYERS explicitly in that case.

BERTSCORE_NUM_LAYERS: int = 12
# Number of transformer layers to use for BERTScore embeddings.
# Must be set explicitly when BERTSCORE_MODEL is a local path (bert_score's
# internal model2layers dict only recognises Hub model names).
# microsoft/mdeberta-v3-base has 12 layers.

BERTSCORE_RESCALE_BASELINE: bool = False
# Rescale BERTScore against a language-specific baseline for better comparability
# across languages. Requires bert_score >= 0.3.11.

BERTSCORE_BATCH_SIZE: int = 32
# Number of samples processed per BERTScore call. Reduce if you get CUDA OOM errors.

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: list[str] = [
    "en", "es", "fr", "pt", "it", "ru",
    "ca", "no", "da", "ro", "de", "el",
    "nl", "cs", "sv",
]
# 15 languages matching the multiclinsum2 dataset.
# Language codes follow ISO 639-1. Norwegian uses "no" (folder: nb).

# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

OUTPUTS_DIR: str = "outputs/classic_metrics"
# Root directory for all evaluation reports. Sub-directories are created
# automatically by runner.py as: {OUTPUTS_DIR}/{team_name}/{language}/

# ---------------------------------------------------------------------------
# BARTScore config (optional, --use_bartscore)
# ---------------------------------------------------------------------------

BARTSCORE_MODEL: str = "facebook/bart-large-cnn"
# HuggingFace Hub name or absolute path to a local BART-large-cnn snapshot.
# Downloads ~1.6 GB on first use if not cached locally.

BARTSCORE_DEVICE: str = "cuda"
# "cuda" for GPU, "cpu" as fallback. BARTScore on CPU is very slow.

BARTSCORE_BATCH_SIZE: int = 4
# Samples per BART forward pass. Reduce if CUDA OOM.

BARTSCORE_MAX_LENGTH: int = 1024
# Max tokens for source and target sequences. Sequences longer than this are truncated.

# ---------------------------------------------------------------------------
# SummaC config (optional, --use_summac)
# ---------------------------------------------------------------------------

SUMMAC_DEVICE: str = "cuda"
# "cuda" for GPU, "cpu" as fallback.

SUMMAC_GRANULARITY: str = "sentence"
# NLI granularity: "sentence" (recommended) or "paragraph" or "document".

SUMMAC_CONV_MODEL_PATH: str = ""
# Absolute path to summac_conv_vitc_sent_perc_e.bin for SummaCConv.
# Leave empty to use the summac library's default download location (~/.cache/summac/).
# Example: "/gpfs/projects/bsc88/NLP4BIA/bsc088665/models/summac_conv_vitc_sent_perc_e.bin"
