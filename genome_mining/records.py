from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SequenceWindow:
    chrom: str
    start: int
    end: int
    sequence: str
    strand: str = "+"
    source: str = ""
    species: str = ""
    annotation_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def window_id(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}:{self.strand}"

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MiningHit:
    chrom: str
    start: int
    end: int
    score: float
    evidence: str
    strand: str = "+"
    window_id: str = ""
    sequence: str = ""

    @property
    def bed_score(self) -> int:
        return max(0, min(1000, int(round(self.score * 100))))
