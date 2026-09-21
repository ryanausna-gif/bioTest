from __future__ import annotations

import math
from random import Random

from .genome import Chromosome


def sample_poisson(lam: float, rng: Random) -> int:
    if lam <= 0:
        return 0
    if lam < 30:
        threshold = math.exp(-lam)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= rng.random()
        return count - 1
    # Normal approximation is sufficient for chromosome-scale crossover counts.
    value = int(round(rng.gauss(lam, math.sqrt(lam))))
    return max(0, value)


def crossover_positions(chrom: Chromosome, rng: Random) -> list[int]:
    count = sample_poisson(chrom.length * chrom.recombination_rate, rng)
    if count <= 0:
        return []
    return sorted(rng.randrange(1, chrom.length) for _ in range(count))


def haplotype_source_at(position: int, initial_source: int, crossovers: list[int]) -> int:
    source = initial_source
    for crossover in crossovers:
        if position < crossover:
            break
        source = 1 - source
    return source
