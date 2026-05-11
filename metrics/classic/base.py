from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def score(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        """
        documents: gold references or source texts (metric-dependent)
        summaries: system-generated candidates, same order as documents
        Returns dict mapping key names to per-sample float lists.
        """
