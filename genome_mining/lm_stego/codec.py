from __future__ import annotations

import hashlib
import math
import zlib

from .arithmetic_codec import ContextArithmeticCoder
from .integer_distribution import IntegerDistributionConfig
from .model_adapter import NucleotideProbabilityModel


FRAME_MAGIC = b"L5A1"
FRAME_HEADER_BYTES = 12


def _padding(key: bytes, sample_id: str, length: int) -> bytes:
    if length <= 0:
        return b""
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(
            hashlib.blake2s(
                sample_id.encode("utf-8") + counter.to_bytes(4, "big"),
                key=key,
                person=b"l5-pad",
                digest_size=32,
            ).digest()
        )
        counter += 1
    return bytes(output[:length])


class LMArithmeticCodec:
    """L5 prototype: authenticated packet carried by a model-driven DNA CDF."""

    embedding_mode = "insertion"

    def __init__(
        self,
        key: bytes | None,
        model: NucleotideProbabilityModel,
        *,
        ecc_symbols: int = 0,
        block_bytes: int = 8,
        max_symbols_per_block: int = 256,
        context_bases: int = 4096,
        distribution_config: IntegerDistributionConfig | None = None,
    ) -> None:
        # Local imports avoid a module cycle with the public codec registry.
        from ..stego_codecs import CodecDescription, _require_key

        self.description = CodecDescription(
            name="lm_arithmetic_v1",
            level="L5-prototype",
            description=(
                "Context-conditioned model CDF with exact self-delimiting inverse arithmetic blocks"
            ),
            requires_key=True,
            requires_cover=False,
            nominal_bits_per_base=0.0,
        )
        self.key = _require_key(key, self.description.name)
        self.model = model
        self.ecc_symbols = int(ecc_symbols)
        self.block_bytes = int(block_bytes)
        self.max_symbols_per_block = int(max_symbols_per_block)
        model_limit = int(getattr(model, "context_limit", 0) or 0)
        self.context_bases = min(int(context_bases), model_limit) if model_limit else int(context_bases)
        self.distribution_config = distribution_config or IntegerDistributionConfig()
        self.coder = ContextArithmeticCoder(
            model,
            block_bytes=self.block_bytes,
            max_symbols_per_block=self.max_symbols_per_block,
            distribution_config=self.distribution_config,
        )

    def _framed_length(self, payload_bytes: int) -> int:
        from ..stego_codecs import _rs_encoded_length

        protected = _rs_encoded_length(payload_bytes + 12 + 16 + 5, self.ecc_symbols)
        unpadded = FRAME_HEADER_BYTES + protected
        return math.ceil(unpadded / self.block_bytes) * self.block_bytes

    def max_encoded_bases(self, payload_bytes: int) -> int:
        blocks = self._framed_length(payload_bytes) // self.block_bytes
        return blocks * self.max_symbols_per_block

    def _frame(self, package: bytes, sample_id: str) -> bytes:
        header = FRAME_MAGIC + len(package).to_bytes(4, "big") + zlib.crc32(package).to_bytes(4, "big")
        value = header + package
        pad_length = (-len(value)) % self.block_bytes
        return value + _padding(self.key, sample_id, pad_length)

    def _unframe(self, value: bytes) -> bytes:
        if len(value) < FRAME_HEADER_BYTES or value[:4] != FRAME_MAGIC:
            raise ValueError("L5 arithmetic frame marker is missing; context, span, or model is wrong.")
        length = int.from_bytes(value[4:8], "big")
        expected_crc = int.from_bytes(value[8:12], "big")
        if length < 1 or FRAME_HEADER_BYTES + length > len(value):
            raise ValueError("L5 arithmetic frame length is invalid.")
        package = value[FRAME_HEADER_BYTES : FRAME_HEADER_BYTES + length]
        if zlib.crc32(package) != expected_crc:
            raise ValueError("L5 arithmetic frame CRC failed.")
        return package

    def encode(
        self,
        payload: bytes,
        *,
        cover: str = "",
        sample_id: str = "",
        context: str = "",
    ) -> str:
        from ..stego_codecs import _rs_encode, _secure_pack

        package = _rs_encode(_secure_pack(payload, self.key, sample_id), self.ecc_symbols)
        framed = self._frame(package, sample_id)
        return self.coder.encode(framed, context=context[-self.context_bases :]).dna

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        from ..stego_codecs import _rs_decode, _secure_unpack

        framed = self.coder.decode(sequence, context=context[-self.context_bases :])
        package = _rs_decode(self._unframe(framed), self.ecc_symbols)
        return _secure_unpack(package, self.key, sample_id, expected_bytes)

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "codec": self.description.name,
            "model": self.model.metadata(),
            "block_bytes": self.block_bytes,
            "max_symbols_per_block": self.max_symbols_per_block,
            "context_bases": self.context_bases,
            "integer_distribution": self.distribution_config.to_dict(),
            "frame_magic": FRAME_MAGIC.decode("ascii"),
            "limitations": [
                "The first implementation targets exact no-noise round trips.",
                "A DNA substitution can desynchronize the rolling language-model context.",
                "The Hugging Face causal adapter requires a checkpoint that exposes next-token logits.",
            ],
        }
