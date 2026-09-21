from __future__ import annotations

from dataclasses import dataclass
from random import Random


BASES = "ACGT"


@dataclass(frozen=True)
class MutationStats:
    snv: int = 0
    insertion: int = 0
    deletion: int = 0

    @property
    def total(self) -> int:
        return self.snv + self.insertion + self.deletion


def mutate_dna(
    dna: str,
    rng: Random,
    snv_rate: float,
    indel_rate: float,
    multiplier: float = 1.0,
) -> tuple[str, MutationStats]:
    snv_p = max(0.0, snv_rate * multiplier)
    indel_p = max(0.0, indel_rate * multiplier)
    out: list[str] = []
    snv = 0
    insertion = 0
    deletion = 0

    for base in dna:
        if rng.random() < indel_p:
            deletion += 1
            continue
        if rng.random() < snv_p:
            choices = [item for item in BASES if item != base]
            base = choices[rng.randrange(len(choices))]
            snv += 1
        out.append(base)
        if rng.random() < indel_p:
            out.append(BASES[rng.randrange(4)])
            insertion += 1

    return "".join(out), MutationStats(snv=snv, insertion=insertion, deletion=deletion)
