"""DNA adaptation of rotation range coding (ACL 2026, Algorithms 3/4).

Independent implementation: exact rational intervals, integer DNA CDF and
domain-separated HMAC offsets instead of the authors' Decimal/Python PRNG.
These changes and message-dependent stopping require separate security analysis.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace
from fractions import Fraction

from .integer_distribution import IntegerDistributionConfig, probabilities_to_integer_distribution

ALPHABET = "ACGT"
PROTOCOL = "dna_rrc_fraction_v1"


class RRCBudgetError(ValueError):
    """A valid message could not fit; distinct from a model/protocol failure."""


@dataclass(frozen=True)
class RRCConfig:
    precision_bits: int = 12
    context_bases: int = 1024
    max_bases: int = 4096

    def __post_init__(self):
        IntegerDistributionConfig(precision_bits=self.precision_bits, uniform_mix=0.0)
        if not 1 <= self.context_bases <= 1048576 or not 1 <= self.max_bases <= 16384:
            raise ValueError("Invalid context/max_bases bounds")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def validate_key_nonce(key, nonce):
    if len(key) < 32 or len(nonce) != 16:
        raise ValueError("Use a key of at least 32 bytes and a fresh 16-byte nonce")


def random_fraction(key, nonce, step, purpose):
    value = hmac.new(key, PROTOCOL.encode() + b"/" + purpose + nonce
                     + step.to_bytes(8, "big"), hashlib.sha256).digest()
    return Fraction(int.from_bytes(value[:16], "big"), 1 << 128)


def clean_context(context, limit):
    context = context.upper()
    if any(base not in "ACGTN" for base in context):
        raise ValueError("Context must contain only A/C/G/T/N")
    return context[-limit:]


def distribution(model, context, config):
    return probabilities_to_integer_distribution(
        model.probabilities(context),
        IntegerDistributionConfig(precision_bits=config.precision_bits, uniform_mix=0.0),
    )


def reverse_value(low, high, history):
    point = (low + high) / 2
    for start, width, offset in reversed(history):
        point = start + (point - start - offset * width) % width
    # Nearest integer with exact half ties toward the lower integer.
    shifted = point - Fraction(1, 2)
    return -(-shifted.numerator // shifted.denominator)


def narrow(low, high, dist, index):
    width = high - low
    return (low + width * Fraction(dist.cumulative[index], dist.total),
            low + width * Fraction(dist.cumulative[index + 1], dist.total))


def encode_bits(bits, model, *, key, nonce, context="", config=None):
    config = config or RRCConfig()
    validate_key_nonce(key, nonce)
    if not 1 <= len(bits) <= 2048 or set(bits) - {"0", "1"}:
        raise ValueError("Message must contain 1..2048 binary digits")
    context = clean_context(context, config.context_bases)
    source = int(bits, 2)
    point = Fraction(source)
    low, high = Fraction(0), Fraction(1 << len(bits))
    history, dna = [], []
    trace = hashlib.sha256()
    entropy, surprisal, rejected_stops = 0.0, 0.0, 0
    for step in range(config.max_bases):
        dist = distribution(model, context, config)
        trace.update(canonical(dist.frequencies))
        offset = random_fraction(key, nonce, step, b"rotation")
        width = high - low
        history.append((low, width, offset))
        point = low + (point - low + offset * width) % width
        ticket = (point - low) * dist.total / width
        index = next(i for i in range(4) if ticket < dist.cumulative[i + 1])
        low, high = narrow(low, high, dist, index)
        base = ALPHABET[index]
        dna.append(base)
        context = (context + base)[-config.context_bases:]
        entropy += -sum((f / dist.total) * math.log2(f / dist.total) for f in dist.frequencies)
        surprisal -= math.log2(dist.frequencies[index] / dist.total)
        if -Fraction(1, 2) < (low + high) / 2 - point <= Fraction(1, 2):
            if reverse_value(low, high, history) == source:
                return "".join(dna), {
                    "message_bits": len(bits), "encoded_bases": len(dna),
                    "bits_per_base": len(bits) / len(dna),
                    "cdf_sha256": trace.hexdigest(), "entropy_sum": entropy,
                    "surprisal_bits": surprisal, "rejected_stops": rejected_stops,
                    "entropy_utilization": len(bits) / entropy,
                }
            rejected_stops += 1
    raise RRCBudgetError("RRC did not finish within max_bases; no successful carrier emitted")


def decode_bits(dna, bit_length, model, *, key, nonce, context="", config=None,
                expected_cdf=None):
    config = config or RRCConfig()
    validate_key_nonce(key, nonce)
    if not 1 <= bit_length <= 2048 or not 1 <= len(dna) <= config.max_bases:
        raise ValueError("Invalid bit length or DNA length")
    if set(dna) - set(ALPHABET):
        raise ValueError("Carrier must contain only uppercase A/C/G/T")
    context = clean_context(context, config.context_bases)
    low, high = Fraction(0), Fraction(1 << bit_length)
    history, trace = [], hashlib.sha256()
    for step, base in enumerate(dna):
        dist = distribution(model, context, config)
        trace.update(canonical(dist.frequencies))
        history.append((low, high - low, random_fraction(key, nonce, step, b"rotation")))
        low, high = narrow(low, high, dist, ALPHABET.index(base))
        context = (context + base)[-config.context_bases:]
    if expected_cdf is not None and not hmac.compare_digest(expected_cdf, trace.hexdigest()):
        raise ValueError("CDF trace mismatch: model/context/numerical environment differs")
    value = reverse_value(low, high, history)
    if not 0 <= value < 1 << bit_length:
        raise ValueError("Recovered value out of range")
    return format(value, f"0{bit_length}b")


def sample_dna(length, model, *, key, nonce, context="", config=None, purpose=b"ordinary"):
    config = config or RRCConfig()
    validate_key_nonce(key, nonce)
    if not 0 <= length <= config.max_bases:
        raise ValueError("Invalid sampling length")
    context = clean_context(context, config.context_bases)
    result = []
    for step in range(length):
        dist = distribution(model, context, config)
        ticket = random_fraction(key, nonce, step, purpose) * dist.total
        index = next(i for i in range(4) if ticket < dist.cumulative[i + 1])
        result.append(ALPHABET[index])
        context = (context + result[-1])[-config.context_bases:]
    return "".join(result)


def encode_packet(bits, model, *, key, nonce, context="", config=None, output_bases=None):
    config = config or RRCConfig()
    context = clean_context(context, config.context_bases)
    if output_bases is not None and not 1 <= output_bases <= config.max_bases:
        raise ValueError("output_bases must be positive and <= max_bases")
    bounded = replace(config, max_bases=output_bases) if output_bases is not None else config
    dna, metrics = encode_bits(bits, model, key=key, nonce=nonce, context=context, config=bounded)
    count = len(dna)
    if output_bases is not None:
        if not count <= output_bases <= config.max_bases:
            raise RRCBudgetError("Carrier exceeds fixed output budget; increase output_bases")
        dna += sample_dna(output_bases - count, model, key=key, nonce=nonce,
                          context=(context + dna)[-config.context_bases:],
                          config=config, purpose=b"padding")
    metadata = {"protocol": PROTOCOL, "config": asdict(config), "nonce": nonce.hex(),
                "context": context, "model": model.metadata(), "metrics": metrics,
                "output_bases": len(dna)}
    metadata["auth_tag"] = hmac.new(key, b"rrc-packet/" + canonical(metadata) + dna.encode(),
                                     hashlib.sha256).hexdigest()
    return dna, metadata


def decode_packet(dna, metadata, model, *, key):
    meta = dict(metadata)
    tag = meta.pop("auth_tag", "")
    expected = hmac.new(key, b"rrc-packet/" + canonical(meta) + dna.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Packet authentication failed: wrong key, DNA or metadata")
    if meta["protocol"] != PROTOCOL or canonical(meta["model"]) != canonical(model.metadata()):
        raise ValueError("Protocol/model mismatch")
    if len(dna) != meta["output_bases"]:
        raise ValueError("Output length mismatch")
    config = RRCConfig(**meta["config"])
    metrics = meta["metrics"]
    return decode_bits(dna[:metrics["encoded_bases"]], metrics["message_bits"], model,
                       key=key, nonce=bytes.fromhex(meta["nonce"]), context=meta["context"],
                       config=config, expected_cdf=metrics["cdf_sha256"])
