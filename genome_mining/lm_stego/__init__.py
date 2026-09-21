"""Context-conditioned generative DNA steganography primitives.

The package keeps the reversible coding core independent from PyTorch. A
probability model only needs to return P(A), P(C), P(G), and P(T) for the next
base. This makes the arithmetic protocol testable with a deterministic k-mer
model before a large DNA language model is used on the server.
"""

from .arithmetic_codec import (
    ArithmeticCodingError,
    ArithmeticDecodeError,
    ArithmeticEncodeResult,
    ContextArithmeticCoder,
)
from .integer_distribution import IntegerDistribution, IntegerDistributionConfig
from .model_adapter import (
    HuggingFaceCausalDNAProbabilityModel,
    KmerProbabilityModel,
    NucleotideProbabilityModel,
    load_probability_model,
)

__all__ = [
    "ArithmeticCodingError",
    "ArithmeticDecodeError",
    "ArithmeticEncodeResult",
    "ContextArithmeticCoder",
    "HuggingFaceCausalDNAProbabilityModel",
    "IntegerDistribution",
    "IntegerDistributionConfig",
    "KmerProbabilityModel",
    "NucleotideProbabilityModel",
    "load_probability_model",
]
