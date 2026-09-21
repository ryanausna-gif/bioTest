from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from itertools import accumulate
from typing import Sequence


@dataclass(frozen=True)
class IntegerDistributionConfig:
    """Rules that turn model probabilities into a reproducible integer CDF."""

    precision_bits: int = 12
    minimum_frequency: int = 1
    uniform_mix: float = 0.20
    rounding_digits: int = 10

    def __post_init__(self) -> None:
        if not 4 <= self.precision_bits <= 24:
            raise ValueError("precision_bits must be between 4 and 24.")
        if self.minimum_frequency < 1:
            raise ValueError("minimum_frequency must be positive.")
        if not 0.0 <= self.uniform_mix < 1.0:
            raise ValueError("uniform_mix must be in [0, 1).")
        if not 3 <= self.rounding_digits <= 15:
            raise ValueError("rounding_digits must be between 3 and 15.")
        if 4 * self.minimum_frequency >= (1 << self.precision_bits):
            raise ValueError("The integer CDF has no mass left after minimum frequencies.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class IntegerDistribution:
    frequencies: tuple[int, int, int, int]
    cumulative: tuple[int, int, int, int, int]
    total: int

    def __post_init__(self) -> None:
        if len(self.frequencies) != 4 or len(self.cumulative) != 5:
            raise ValueError("A DNA distribution must contain four symbols.")
        if any(value <= 0 for value in self.frequencies):
            raise ValueError("All DNA symbols need positive integer frequency.")
        if self.cumulative[0] != 0 or self.cumulative[-1] != self.total:
            raise ValueError("Invalid cumulative distribution bounds.")
        if tuple(accumulate((0, *self.frequencies))) != self.cumulative:
            raise ValueError("Cumulative frequencies do not match symbol frequencies.")

    def probability(self, symbol_index: int) -> float:
        return self.frequencies[symbol_index] / self.total


def probabilities_to_integer_distribution(
    probabilities: Sequence[float],
    config: IntegerDistributionConfig | None = None,
) -> IntegerDistribution:
    """Quantize four probabilities with deterministic largest-remainder rounding.

    Model logits can differ in their last floating-point bits across devices.
    Values are rounded to a declared decimal precision before allocation. The
    resulting integer CDF is part of the L5 wire protocol and must therefore be
    identical at encoding and decoding time.
    """

    cfg = config or IntegerDistributionConfig()
    if len(probabilities) != 4:
        raise ValueError("Expected P(A), P(C), P(G), and P(T).")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in probabilities):
        raise ValueError("Model probabilities must be finite and non-negative.")

    rounded = [round(float(value), cfg.rounding_digits) for value in probabilities]
    if sum(rounded) <= 0.0:
        raise ValueError("At least one model probability must be positive.")

    with localcontext() as decimal_context:
        decimal_context.prec = max(28, cfg.rounding_digits + 16)
        values = [Decimal(str(value)) for value in rounded]
        normalizer = sum(values)
        values = [value / normalizer for value in values]
        mix = Decimal(str(cfg.uniform_mix))
        quarter = Decimal(1) / Decimal(4)
        values = [(Decimal(1) - mix) * value + mix * quarter for value in values]
        normalizer = sum(values)
        values = [value / normalizer for value in values]

        total = 1 << cfg.precision_bits
        remaining = total - 4 * cfg.minimum_frequency
        raw = [value * remaining for value in values]
        floors = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in raw]
        leftover = remaining - sum(floors)
        order = sorted(
            range(4),
            key=lambda index: (-(raw[index] - floors[index]), index),
        )
        for index in order[:leftover]:
            floors[index] += 1

    frequencies = tuple(value + cfg.minimum_frequency for value in floors)
    cumulative = tuple(accumulate((0, *frequencies)))
    return IntegerDistribution(
        frequencies=frequencies,  # type: ignore[arg-type]
        cumulative=cumulative,  # type: ignore[arg-type]
        total=total,
    )
