from __future__ import annotations

import hashlib
import itertools
import math
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DNA_ALPHABET = "ACGT"
ATCG_PAIR_TO_BASE = ("A", "T", "C", "G")
ATCG_BASE_TO_PAIR = {base: value for value, base in enumerate(ATCG_PAIR_TO_BASE)}


@dataclass(frozen=True)
class CodecDescription:
    name: str
    level: str
    description: str
    requires_key: bool
    requires_cover: bool
    nominal_bits_per_base: float


class ImageStegoCodec(ABC):
    description: CodecDescription
    embedding_mode = "insertion"
    context_bases = 0

    @abstractmethod
    def max_encoded_bases(self, payload_bytes: int) -> int:
        """Return a conservative upper bound used to choose a legal insertion point."""

    @abstractmethod
    def encode(
        self,
        payload: bytes,
        *,
        cover: str = "",
        sample_id: str = "",
        context: str = "",
    ) -> str:
        """Encode image bytes to a reversible DNA payload."""

    @abstractmethod
    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        """Recover the original image bytes from an exact payload span."""


def load_codec_key(path: str | Path | None) -> bytes | None:
    if path is None:
        return None
    value = Path(path).read_bytes()
    if not value:
        raise ValueError(f"Codec key file is empty: {path}")
    try:
        text = value.decode("ascii").strip()
        if text.startswith("hex:"):
            return bytes.fromhex(text[4:].strip())
    except (UnicodeDecodeError, ValueError):
        pass
    return value


def codec_key_fingerprint(key: bytes | None) -> str:
    return "" if key is None else hashlib.sha256(key).hexdigest()[:16]


def _require_key(key: bytes | None, codec_name: str) -> bytes:
    if key is None or len(key) < 16:
        raise ValueError(f"{codec_name} requires a codec key containing at least 16 bytes.")
    # Normalize arbitrary key-file lengths to the 32-byte limit accepted by BLAKE2s and ChaCha20.
    return hashlib.sha256(key).digest()


def _derive_key(master_key: bytes, sample_id: str, purpose: bytes) -> bytes:
    return hashlib.blake2b(
        sample_id.encode("utf-8"),
        key=master_key,
        person=purpose[:16],
        digest_size=32,
    ).digest()


def _secure_pack(payload: bytes, master_key: bytes, sample_id: str) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as exc:
        raise ImportError(
            "Encrypted stego codecs require cryptography. Install requirements-hyenadna-stego.txt."
        ) from exc

    compressed = zlib.compress(payload, level=9)
    use_compressed = len(compressed) < len(payload)
    body = compressed if use_compressed else payload
    plaintext = bytes([int(use_compressed)]) + len(payload).to_bytes(4, "big") + body
    key = _derive_key(master_key, sample_id, b"dna-stego-aead")
    nonce = hashlib.blake2s(
        sample_id.encode("utf-8") + hashlib.sha256(payload).digest(),
        key=master_key,
        person=b"dnanonce",
        digest_size=12,
    ).digest()
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, sample_id.encode("utf-8"))
    return nonce + ciphertext


def _secure_unpack(package: bytes, master_key: bytes, sample_id: str, expected_bytes: int) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from cryptography.exceptions import InvalidTag
    except ImportError as exc:
        raise ImportError(
            "Encrypted stego codecs require cryptography. Install requirements-hyenadna-stego.txt."
        ) from exc

    if len(package) < 12 + 16 + 5:
        raise ValueError("Encrypted DNA package is truncated.")
    nonce, ciphertext = package[:12], package[12:]
    key = _derive_key(master_key, sample_id, b"dna-stego-aead")
    try:
        plaintext = ChaCha20Poly1305(key).decrypt(
            nonce, ciphertext, sample_id.encode("utf-8")
        )
    except InvalidTag as exc:
        raise ValueError("Encrypted DNA authentication failed; the span, key, or sequence is wrong.") from exc
    compressed = bool(plaintext[0])
    declared_length = int.from_bytes(plaintext[1:5], "big")
    body = plaintext[5:]
    try:
        payload = zlib.decompress(body) if compressed else body
    except zlib.error as exc:
        raise ValueError("Compressed image payload is damaged.") from exc
    if declared_length != expected_bytes or len(payload) != expected_bytes:
        raise ValueError(
            f"Decoded image has {len(payload)} bytes and declares {declared_length}; "
            f"expected {expected_bytes}."
        )
    return payload


def _rs_encoded_length(data_bytes: int, ecc_symbols: int) -> int:
    if not ecc_symbols:
        return data_bytes
    if not 1 <= ecc_symbols < 255:
        raise ValueError("ecc_symbols must be between one and 254.")
    data_per_block = 255 - ecc_symbols
    return data_bytes + math.ceil(data_bytes / data_per_block) * ecc_symbols


def _rs_encode(payload: bytes, ecc_symbols: int) -> bytes:
    if not ecc_symbols:
        return payload
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise ImportError("Reed-Solomon protection requires reedsolo.") from exc
    return bytes(RSCodec(ecc_symbols).encode(payload))


def _rs_decode(payload: bytes, ecc_symbols: int) -> bytes:
    if not ecc_symbols:
        return payload
    try:
        from reedsolo import RSCodec, ReedSolomonError
    except ImportError as exc:
        raise ImportError("Reed-Solomon protection requires reedsolo.") from exc
    try:
        decoded = RSCodec(ecc_symbols).decode(payload)
    except ReedSolomonError as exc:
        raise ValueError("Reed-Solomon decoding failed.") from exc
    return bytes(decoded[0] if isinstance(decoded, tuple) else decoded)


def _bytes_to_atcg(payload: bytes) -> str:
    return "".join(
        ATCG_PAIR_TO_BASE[(value >> shift) & 0b11]
        for value in payload
        for shift in (6, 4, 2, 0)
    )


def _atcg_to_bytes(sequence: str) -> bytes:
    dna = str(sequence).upper()
    if len(dna) % 4:
        raise ValueError("Direct 2-bit DNA length must be divisible by four.")
    try:
        pairs = [ATCG_BASE_TO_PAIR[base] for base in dna]
    except KeyError as exc:
        raise ValueError("Direct 2-bit DNA may contain only A/C/G/T.") from exc
    output = bytearray()
    for offset in range(0, len(pairs), 4):
        value = 0
        for pair in pairs[offset : offset + 4]:
            value = (value << 2) | pair
        output.append(value)
    return bytes(output)


class DirectATCGCodec(ImageStegoCodec):
    description = CodecDescription(
        name="direct_2bit_atcg_v1",
        level="L0",
        description="00=A, 01=T, 10=C, 11=G direct image-byte mapping",
        requires_key=False,
        requires_cover=False,
        nominal_bits_per_base=2.0,
    )

    def __init__(self, *, ecc_symbols: int = 0) -> None:
        self.ecc_symbols = int(ecc_symbols)

    def max_encoded_bases(self, payload_bytes: int) -> int:
        return _rs_encoded_length(payload_bytes, self.ecc_symbols) * 4

    def encode(
        self, payload: bytes, *, cover: str = "", sample_id: str = "", context: str = ""
    ) -> str:
        return _bytes_to_atcg(_rs_encode(payload, self.ecc_symbols))

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        payload = _rs_decode(_atcg_to_bytes(sequence), self.ecc_symbols)
        if len(payload) != expected_bytes:
            raise ValueError(f"Decoded {len(payload)} bytes; expected {expected_bytes}.")
        return payload


class LegacyRawACGTCodec(ImageStegoCodec):
    description = CodecDescription(
        name="raw_gray_2bit_v1",
        level="L0-legacy",
        description="Legacy A=00, C=01, G=10, T=11 direct mapping",
        requires_key=False,
        requires_cover=False,
        nominal_bits_per_base=2.0,
    )

    def max_encoded_bases(self, payload_bytes: int) -> int:
        return payload_bytes * 4

    def encode(
        self, payload: bytes, *, cover: str = "", sample_id: str = "", context: str = ""
    ) -> str:
        from .image_payload import bytes_to_dna

        return bytes_to_dna(payload)

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        from .image_payload import dna_to_bytes

        return dna_to_bytes(sequence, expected_bytes=expected_bytes)


class EncryptedATCGCodec(ImageStegoCodec):
    description = CodecDescription(
        name="encrypted_2bit_atcg_v1",
        level="L1",
        description="Compression plus ChaCha20-Poly1305 followed by direct 2-bit mapping",
        requires_key=True,
        requires_cover=False,
        nominal_bits_per_base=2.0,
    )

    def __init__(self, key: bytes | None, *, ecc_symbols: int = 0) -> None:
        self.key = _require_key(key, self.description.name)
        self.ecc_symbols = int(ecc_symbols)

    def max_encoded_bases(self, payload_bytes: int) -> int:
        # nonce + AEAD tag + private five-byte framing; compression can only reduce the body
        return _rs_encoded_length(payload_bytes + 12 + 16 + 5, self.ecc_symbols) * 4

    def encode(
        self, payload: bytes, *, cover: str = "", sample_id: str = "", context: str = ""
    ) -> str:
        package = _rs_encode(_secure_pack(payload, self.key, sample_id), self.ecc_symbols)
        return _bytes_to_atcg(package)

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        package = _rs_decode(_atcg_to_bytes(sequence), self.ecc_symbols)
        return _secure_unpack(package, self.key, sample_id, expected_bytes)


def _max_homopolymer(sequence: str) -> int:
    longest = current = 0
    previous = ""
    for base in sequence:
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
        previous = base
    return longest


def _valid_six_base_codewords() -> tuple[str, ...]:
    words = []
    for values in itertools.product(DNA_ALPHABET, repeat=6):
        word = "".join(values)
        gc = word.count("G") + word.count("C")
        if 2 <= gc <= 4 and _max_homopolymer(word) <= 3:
            words.append(word)
    return tuple(words)


_SIX_BASE_CODEWORDS = _valid_six_base_codewords()


class ConstrainedBlockCodec(ImageStegoCodec):
    description = CodecDescription(
        name="constrained_block_v1",
        level="L2",
        description="Encrypted bytes mapped to constrained six-base codewords",
        requires_key=True,
        requires_cover=False,
        nominal_bits_per_base=8.0 / 6.0,
    )

    def __init__(
        self,
        key: bytes | None,
        *,
        match_cover: bool = False,
        kmer_order: int = 3,
        ecc_symbols: int = 0,
    ) -> None:
        self.key = _require_key(key, self.description.name)
        self.ecc_symbols = int(ecc_symbols)
        if kmer_order < 2 or kmer_order > 6:
            raise ValueError("kmer_order must be between two and six.")
        self.match_cover = bool(match_cover)
        self.kmer_order = int(kmer_order)
        if self.match_cover:
            self.description = CodecDescription(
                name="cover_kmer_block_v1",
                level="L3",
                description="Encrypted constrained codewords selected to match their genomic cover",
                requires_key=True,
                requires_cover=True,
                nominal_bits_per_base=8.0 / 6.0,
            )
        ordered = sorted(
            _SIX_BASE_CODEWORDS,
            key=lambda word: hashlib.blake2s(
                word.encode("ascii"), key=self.key, person=b"dna-code", digest_size=8
            ).digest(),
        )
        groups: list[list[str]] = [[] for _ in range(256)]
        decoder: dict[str, int] = {}
        for index, word in enumerate(ordered):
            value = index % 256
            groups[value].append(word)
            decoder[word] = value
        if min(map(len, groups)) < 2:
            raise AssertionError("The constrained codebook must provide multiple words per byte.")
        self.groups = tuple(tuple(group) for group in groups)
        self.decoder = decoder

    def max_encoded_bases(self, payload_bytes: int) -> int:
        return _rs_encoded_length(payload_bytes + 12 + 16 + 5, self.ecc_symbols) * 6

    def _cover_kmers(self, cover: str) -> tuple[Counter[str], int]:
        order = self.kmer_order
        counts = Counter(cover[index : index + order] for index in range(len(cover) - order + 1))
        return counts, sum(counts.values())

    def _candidate_score(
        self,
        word: str,
        target: str,
        prefix: str,
        counts: Counter[str],
        total_kmers: int,
        tie_material: bytes,
    ) -> tuple[float, bytes]:
        hamming = sum(left != right for left, right in zip(word, target))
        gc_delta = abs((word.count("G") + word.count("C")) - (target.count("G") + target.count("C")))
        context = prefix[-(self.kmer_order - 1) :] + word
        kmer_penalty = 0.0
        vocabulary = 4**self.kmer_order
        for index in range(len(context) - self.kmer_order + 1):
            kmer = context[index : index + self.kmer_order]
            probability = (counts.get(kmer, 0) + 1.0) / (total_kmers + vocabulary)
            kmer_penalty -= math.log(probability)
        run_penalty = max(0, _max_homopolymer(prefix[-3:] + word) - 3) * 10.0
        score = hamming + 0.75 * gc_delta + 0.20 * kmer_penalty + run_penalty
        tie = hashlib.blake2s(word.encode("ascii") + tie_material, key=self.key, digest_size=8).digest()
        return score, tie

    def encode(
        self, payload: bytes, *, cover: str = "", sample_id: str = "", context: str = ""
    ) -> str:
        package = _rs_encode(_secure_pack(payload, self.key, sample_id), self.ecc_symbols)
        required = len(package) * 6
        if self.match_cover and len(cover) < required:
            raise ValueError(f"cover_kmer_block_v1 needs {required} cover bases; received {len(cover)}.")
        counts, total = self._cover_kmers(cover) if self.match_cover else (Counter(), 0)
        output: list[str] = []
        for index, value in enumerate(package):
            candidates = self.groups[value]
            if self.match_cover:
                target = cover[index * 6 : index * 6 + 6]
                prefix = "".join(output[-1:])
                tie = sample_id.encode("utf-8") + index.to_bytes(8, "big")
                word = min(
                    candidates,
                    key=lambda candidate: self._candidate_score(
                        candidate, target, prefix, counts, total, tie
                    ),
                )
            else:
                digest = hashlib.blake2s(
                    sample_id.encode("utf-8") + index.to_bytes(8, "big"),
                    key=self.key,
                    person=b"dna-pick",
                    digest_size=4,
                ).digest()
                word = candidates[int.from_bytes(digest, "big") % len(candidates)]
            output.append(word)
        return "".join(output)

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        dna = str(sequence).upper()
        if len(dna) % 6:
            raise ValueError("Constrained block DNA length must be divisible by six.")
        decoded_values = []
        for index in range(0, len(dna), 6):
            word = dna[index : index + 6]
            value = self.decoder.get(word)
            if value is None:
                if any(base not in DNA_ALPHABET for base in word):
                    raise ValueError("Constrained DNA contains a non-ACGT base.")
                # Convert an invalid constrained word to its nearest legal word.
                # The downstream RS/authentication layers decide whether the
                # resulting byte stream is recoverable rather than silently accepting it.
                nearest = min(
                    _SIX_BASE_CODEWORDS,
                    key=lambda candidate: (
                        sum(left != right for left, right in zip(word, candidate)),
                        candidate,
                    ),
                )
                value = self.decoder[nearest]
            decoded_values.append(value)
        package = bytes(decoded_values)
        package = _rs_decode(package, self.ecc_symbols)
        return _secure_unpack(package, self.key, sample_id, expected_bytes)


_BIT_ZERO = frozenset("AC")
_GC_PRESERVING_FLIP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _cover_bit(base: str) -> int:
    if base not in DNA_ALPHABET:
        raise ValueError("Cover-Hamming coding requires an A/C/G/T-only cover.")
    return int(base not in _BIT_ZERO)


class CoverHammingCodec(ImageStegoCodec):
    embedding_mode = "replacement"
    description = CodecDescription(
        name="cover_hamming_v1",
        level="L4",
        description="Hamming matrix embedding: three encrypted bits per seven cover bases",
        requires_key=True,
        requires_cover=True,
        nominal_bits_per_base=3.0 / 7.0,
    )

    def __init__(self, key: bytes | None, *, ecc_symbols: int = 0) -> None:
        self.key = _require_key(key, self.description.name)
        self.ecc_symbols = int(ecc_symbols)

    def max_encoded_bases(self, payload_bytes: int) -> int:
        package_bytes = _rs_encoded_length(payload_bytes + 12 + 16 + 5, self.ecc_symbols)
        return math.ceil((package_bytes * 8) / 3) * 7

    def encode(
        self, payload: bytes, *, cover: str = "", sample_id: str = "", context: str = ""
    ) -> str:
        package = _rs_encode(_secure_pack(payload, self.key, sample_id), self.ecc_symbols)
        bit_string = "".join(f"{value:08b}" for value in package)
        bit_string += "0" * ((-len(bit_string)) % 3)
        required = (len(bit_string) // 3) * 7
        if len(cover) < required:
            raise ValueError(f"cover_hamming_v1 needs {required} cover bases; received {len(cover)}.")
        output = list(cover[:required].upper())
        for group_index in range(len(bit_string) // 3):
            start = group_index * 7
            group = output[start : start + 7]
            syndrome = 0
            for position, base in enumerate(group, start=1):
                if _cover_bit(base):
                    syndrome ^= position
            desired = int(bit_string[group_index * 3 : group_index * 3 + 3], 2)
            flip_position = syndrome ^ desired
            if flip_position:
                absolute = start + flip_position - 1
                output[absolute] = _GC_PRESERVING_FLIP[output[absolute]]
        return "".join(output)

    def decode(
        self,
        sequence: str,
        *,
        expected_bytes: int,
        sample_id: str = "",
        context: str = "",
    ) -> bytes:
        dna = str(sequence).upper()
        if len(dna) % 7:
            raise ValueError("Cover-Hamming DNA length must be divisible by seven.")
        bits: list[str] = []
        for start in range(0, len(dna), 7):
            syndrome = 0
            for position, base in enumerate(dna[start : start + 7], start=1):
                if _cover_bit(base):
                    syndrome ^= position
            bits.append(f"{syndrome:03b}")
        bit_string = "".join(bits)
        package = bytes(
            int(bit_string[index : index + 8], 2)
            for index in range(0, len(bit_string) - 7, 8)
        )
        package = _rs_decode(package, self.ecc_symbols)
        return _secure_unpack(package, self.key, sample_id, expected_bytes)


CODEC_ALIASES = {
    "direct": "direct_2bit_atcg_v1",
    "l0": "direct_2bit_atcg_v1",
    "encrypted": "encrypted_2bit_atcg_v1",
    "l1": "encrypted_2bit_atcg_v1",
    "constrained": "constrained_block_v1",
    "l2": "constrained_block_v1",
    "kmer": "cover_kmer_block_v1",
    "l3": "cover_kmer_block_v1",
    "cover": "cover_hamming_v1",
    "l4": "cover_hamming_v1",
    "lm": "lm_arithmetic_v1",
    "l5": "lm_arithmetic_v1",
    "arithmetic": "lm_arithmetic_v1",
}


def normalize_codec_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return CODEC_ALIASES.get(normalized, normalized)


def make_stego_codec(
    name: str,
    *,
    key: bytes | None = None,
    kmer_order: int = 3,
    ecc_symbols: int = 0,
    lm_model=None,
    lm_model_path: str | Path | None = None,
    lm_block_bytes: int = 8,
    lm_max_symbols_per_block: int = 256,
    lm_context_bases: int = 4096,
    lm_cdf_precision_bits: int = 12,
    lm_uniform_mix: float = 0.20,
) -> ImageStegoCodec:
    normalized = normalize_codec_name(name)
    if normalized == "direct_2bit_atcg_v1":
        return DirectATCGCodec(ecc_symbols=ecc_symbols)
    if normalized == "raw_gray_2bit_v1":
        return LegacyRawACGTCodec()
    if normalized == "encrypted_2bit_atcg_v1":
        return EncryptedATCGCodec(key, ecc_symbols=ecc_symbols)
    if normalized == "constrained_block_v1":
        return ConstrainedBlockCodec(
            key, match_cover=False, kmer_order=kmer_order, ecc_symbols=ecc_symbols
        )
    if normalized == "cover_kmer_block_v1":
        return ConstrainedBlockCodec(
            key, match_cover=True, kmer_order=kmer_order, ecc_symbols=ecc_symbols
        )
    if normalized == "cover_hamming_v1":
        return CoverHammingCodec(key, ecc_symbols=ecc_symbols)
    if normalized == "lm_arithmetic_v1":
        from .lm_stego.codec import LMArithmeticCodec
        from .lm_stego.integer_distribution import IntegerDistributionConfig
        from .lm_stego.model_adapter import load_probability_model

        if lm_model is None:
            if lm_model_path is None:
                raise ValueError(
                    "lm_arithmetic_v1 requires lm_model or lm_model_path. "
                    "Create one with train-l5-context-model."
                )
            lm_model = load_probability_model(lm_model_path)
        return LMArithmeticCodec(
            key,
            lm_model,
            ecc_symbols=ecc_symbols,
            block_bytes=lm_block_bytes,
            max_symbols_per_block=lm_max_symbols_per_block,
            context_bases=lm_context_bases,
            distribution_config=IntegerDistributionConfig(
                precision_bits=lm_cdf_precision_bits,
                uniform_mix=lm_uniform_mix,
            ),
        )
    raise ValueError(
        f"Unknown stego codec {name!r}. Available: direct, encrypted, constrained, kmer, cover, lm."
    )


def available_stego_codecs() -> list[dict[str, object]]:
    rows = []
    for name in (
        "direct_2bit_atcg_v1",
        "encrypted_2bit_atcg_v1",
        "constrained_block_v1",
        "cover_kmer_block_v1",
        "cover_hamming_v1",
        "lm_arithmetic_v1",
    ):
        if name == "direct_2bit_atcg_v1":
            description = DirectATCGCodec.description
        elif name == "encrypted_2bit_atcg_v1":
            description = EncryptedATCGCodec.description
        elif name == "constrained_block_v1":
            description = ConstrainedBlockCodec.description
        elif name == "cover_kmer_block_v1":
            description = ConstrainedBlockCodec(b"0" * 16, match_cover=True).description
        elif name == "cover_hamming_v1":
            description = CoverHammingCodec.description
        else:
            description = CodecDescription(
                name="lm_arithmetic_v1",
                level="L5-prototype",
                description=(
                    "Context-conditioned model CDF with exact self-delimiting inverse arithmetic blocks"
                ),
                requires_key=True,
                requires_cover=False,
                nominal_bits_per_base=0.0,
            )
        rows.append(description.__dict__)
    return rows
