from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from fractions import Fraction

from .integer_distribution import (
    IntegerDistribution,
    IntegerDistributionConfig,
    probabilities_to_integer_distribution,
)
from .model_adapter import DNA_ALPHABET, NucleotideProbabilityModel


class ArithmeticCodingError(ValueError):
    pass


class ArithmeticDecodeError(ArithmeticCodingError):
    pass


@dataclass(frozen=True)
class ArithmeticEncodeResult:
    dna: str
    source_bytes: int
    source_bits: int
    block_symbols: tuple[int, ...]
    model_bits: float

    @property
    def effective_bits_per_base(self) -> float:
        return self.source_bits / len(self.dna) if self.dna else 0.0

    @property
    def mean_symbols_per_block(self) -> float:
        return sum(self.block_symbols) / len(self.block_symbols) if self.block_symbols else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_bytes": self.source_bytes,
            "source_bits": self.source_bits,
            "encoded_bases": len(self.dna),
            "n_blocks": len(self.block_symbols),
            "block_symbols": list(self.block_symbols),
            "mean_symbols_per_block": self.mean_symbols_per_block,
            "effective_bits_per_base": self.effective_bits_per_base,
            "model_bits": self.model_bits,
        }


def _child_interval(
    low: Fraction,
    high: Fraction,
    distribution: IntegerDistribution,
    symbol_index: int,
) -> tuple[Fraction, Fraction]:
    width = high - low
    child_low = low + width * Fraction(distribution.cumulative[symbol_index], distribution.total)
    child_high = low + width * Fraction(
        distribution.cumulative[symbol_index + 1], distribution.total
    )
    return child_low, child_high


def _symbol_for_point(
    point: Fraction,
    low: Fraction,
    high: Fraction,
    distribution: IntegerDistribution,
) -> int:
    if not low <= point < high:
        raise ArithmeticCodingError("The target point escaped the arithmetic interval.")
    ticket_fraction = (point - low) * distribution.total / (high - low)
    ticket = ticket_fraction.numerator // ticket_fraction.denominator
    if not 0 <= ticket < distribution.total:
        raise ArithmeticCodingError("Invalid integer CDF ticket.")
    return bisect.bisect_right(distribution.cumulative, ticket) - 1


def _contained_cell(low: Fraction, high: Fraction, bits: int) -> int | None:
    scale = 1 << bits
    scaled_low = low * scale
    candidate = scaled_low.numerator // scaled_low.denominator
    if not 0 <= candidate < scale:
        return None
    cell_low = Fraction(candidate, scale)
    cell_high = Fraction(candidate + 1, scale)
    return candidate if low >= cell_low and high <= cell_high else None


class ContextArithmeticCoder:
    """Exact blockwise inverse arithmetic coder over A/C/G/T.

    A uniformly distributed encrypted block selects one binary cell. Bases are
    generated from the model CDF until the sequence interval lies completely
    inside that cell. The decoder detects the same condition, making blocks
    self-delimiting without an explicit DNA marker.
    """

    def __init__(
        self,
        model: NucleotideProbabilityModel,
        *,
        block_bytes: int = 8,
        max_symbols_per_block: int = 256,
        distribution_config: IntegerDistributionConfig | None = None,
    ) -> None:
        if block_bytes <= 0 or block_bytes > 32:
            raise ValueError("block_bytes must be between one and 32.")
        if max_symbols_per_block < block_bytes * 4:
            raise ValueError("max_symbols_per_block is too small for the configured block size.")
        self.model = model
        self.block_bytes = int(block_bytes)
        self.max_symbols_per_block = int(max_symbols_per_block)
        self.distribution_config = distribution_config or IntegerDistributionConfig()

    def _context(self, value: str) -> str:
        limit = int(getattr(self.model, "context_limit", 0) or 0)
        return value[-limit:] if limit else value

    def _distribution(self, context: str) -> IntegerDistribution:
        return probabilities_to_integer_distribution(
            self.model.probabilities(self._context(context)),
            self.distribution_config,
        )

    def encode(self, payload: bytes, *, context: str = "") -> ArithmeticEncodeResult:
        if not payload:
            return ArithmeticEncodeResult("", 0, 0, (), 0.0)
        output: list[str] = []
        block_symbols: list[int] = []
        model_bits = 0.0
        rolling_context = str(context).upper()

        for offset in range(0, len(payload), self.block_bytes):
            block = payload[offset : offset + self.block_bytes]
            bits = len(block) * 8
            scale = 1 << bits
            value = int.from_bytes(block, "big")
            cell_low = Fraction(value, scale)
            cell_high = Fraction(value + 1, scale)
            # One third is rationally separated from all power-of-two CDF
            # boundaries, avoiding exact-boundary ambiguity.
            point = Fraction(3 * value + 1, 3 * scale)
            low, high = Fraction(0), Fraction(1)
            produced = 0

            while produced < self.max_symbols_per_block:
                distribution = self._distribution(rolling_context)
                symbol_index = _symbol_for_point(point, low, high, distribution)
                low, high = _child_interval(low, high, distribution, symbol_index)
                base = DNA_ALPHABET[symbol_index]
                output.append(base)
                rolling_context += base
                produced += 1
                model_bits -= math.log2(distribution.probability(symbol_index))
                if low >= cell_low and high <= cell_high:
                    break
            else:
                raise ArithmeticCodingError(
                    f"A {bits}-bit block did not converge within "
                    f"{self.max_symbols_per_block} bases. Increase uniform_mix or the symbol limit."
                )
            block_symbols.append(produced)

        dna = "".join(output)
        return ArithmeticEncodeResult(
            dna=dna,
            source_bytes=len(payload),
            source_bits=len(payload) * 8,
            block_symbols=tuple(block_symbols),
            model_bits=model_bits,
        )

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int | None = None,
        context: str = "",
    ) -> bytes:
        dna = str(sequence).upper()
        if any(base not in DNA_ALPHABET for base in dna):
            raise ArithmeticDecodeError("L5 arithmetic DNA may contain only A/C/G/T.")
        if expected_bytes is not None and expected_bytes < 0:
            raise ValueError("expected_bytes cannot be negative.")
        if not dna:
            if expected_bytes in (None, 0):
                return b""
            raise ArithmeticDecodeError("The DNA span ended before the first arithmetic block.")

        output = bytearray()
        rolling_context = str(context).upper()
        dna_offset = 0

        while dna_offset < len(dna) and (expected_bytes is None or len(output) < expected_bytes):
            remaining = None if expected_bytes is None else expected_bytes - len(output)
            block_size = self.block_bytes if remaining is None else min(self.block_bytes, remaining)
            bits = block_size * 8
            low, high = Fraction(0), Fraction(1)
            recovered_value = None

            for _ in range(self.max_symbols_per_block):
                if dna_offset >= len(dna):
                    raise ArithmeticDecodeError("The DNA span ended inside an arithmetic block.")
                distribution = self._distribution(rolling_context)
                base = dna[dna_offset]
                symbol_index = DNA_ALPHABET.index(base)
                low, high = _child_interval(low, high, distribution, symbol_index)
                rolling_context += base
                dna_offset += 1
                recovered_value = _contained_cell(low, high, bits)
                if recovered_value is not None:
                    output.extend(recovered_value.to_bytes(block_size, "big"))
                    break
            else:
                raise ArithmeticDecodeError(
                    f"No {bits}-bit cell was identified within "
                    f"{self.max_symbols_per_block} bases."
                )

        if expected_bytes is not None and len(output) != expected_bytes:
            raise ArithmeticDecodeError(
                f"Recovered {len(output)} bytes; expected {expected_bytes}."
            )
        if dna_offset != len(dna):
            raise ArithmeticDecodeError(
                f"The arithmetic payload decoded before {len(dna) - dna_offset} trailing bases."
            )
        return bytes(output)
