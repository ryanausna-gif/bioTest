from __future__ import annotations

import csv
import hashlib
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .fasta import iter_fasta_records
from .features import gc_content, shannon_entropy_from_counts

ALPHABET = "ACGT"
BIT_TO_DNA = [
    {"00": "A", "01": "C", "10": "G", "11": "T"},
    {"00": "C", "01": "A", "10": "T", "11": "G"},
    {"00": "G", "01": "T", "10": "A", "11": "C"},
    {"00": "T", "01": "G", "10": "C", "11": "A"},
]


@dataclass(frozen=True)
class CarrierResult:
    sequence: str
    y: int
    codec_family: str
    codec_rule: str
    carrier_type: str
    message_type: str
    embedding_type: str
    detectability: str
    start: int
    end: int
    strength: float
    background_id: str
    naturalness_label: int
    metadata: dict[str, object]


class FastaBackgroundSampler:
    def __init__(self, fasta_path: str | Path | None = None, length: int = 1000):
        self.length = int(length)
        self.records: list[tuple[str, str]] = []
        if fasta_path:
            for record in iter_fasta_records(fasta_path, min_length=length):
                self.records.append((record.name, record.sequence))

    def sample(self, split: str = "train", protocol: str = "standard") -> tuple[str, str]:
        if self.records:
            name, sequence = random.choice(self.records)
            start = 0 if len(sequence) <= self.length else random.randint(0, len(sequence) - self.length)
            return sequence[start : start + self.length], f"fasta:{name}:{start}-{start + self.length}"
        return synthetic_background(self.length, choose_background_id(split, protocol))


def set_seed(seed: int) -> None:
    random.seed(seed)


def weighted_random_seq(length: int, probs: dict[str, float] | None = None) -> str:
    probs = probs or {base: 0.25 for base in ALPHABET}
    bases = list(ALPHABET)
    weights = [float(probs.get(base, 0.0)) for base in bases]
    total = sum(weights)
    if total <= 0:
        weights = [0.25] * 4
    else:
        weights = [weight / total for weight in weights]
    return "".join(random.choices(bases, weights=weights, k=length))


def synthetic_background(length: int, background_id: str = "balanced") -> tuple[str, str]:
    profiles = {
        "balanced": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        "at_rich": {"A": 0.35, "C": 0.15, "G": 0.15, "T": 0.35},
        "gc_rich": {"A": 0.16, "C": 0.34, "G": 0.34, "T": 0.16},
        "skew_a": {"A": 0.45, "C": 0.20, "G": 0.15, "T": 0.20},
        "skew_t": {"A": 0.20, "C": 0.15, "G": 0.20, "T": 0.45},
    }
    return weighted_random_seq(length, profiles[background_id]), background_id


def choose_background_id(split: str, protocol: str) -> str:
    if protocol == "background_shift":
        return random.choice(["balanced", "at_rich"] if split in {"train", "val"} else ["gc_rich", "skew_a", "skew_t"])
    return random.choice(["balanced", "at_rich", "gc_rich", "skew_a", "skew_t"])


def sequence_entropy(sequence: str) -> float:
    counts: dict[str, int] = {}
    for base in sequence:
        counts[base] = counts.get(base, 0) + 1
    return shannon_entropy_from_counts(counts)  # dict values are sufficient for the entropy helper.


def match_gc(sequence: str, target: float, tol: float = 0.02, max_iter: int = 2000) -> str:
    arr = list(sequence)
    for _ in range(max_iter):
        current = gc_content("".join(arr))
        if abs(current - target) <= tol:
            break
        if current < target:
            idx = [i for i, base in enumerate(arr) if base in "AT"]
            if not idx:
                break
            arr[random.choice(idx)] = random.choice("GC")
        else:
            idx = [i for i, base in enumerate(arr) if base in "GC"]
            if not idx:
                break
            arr[random.choice(idx)] = random.choice("AT")
    return "".join(arr)


def avoid_homopolymer(sequence: str, max_len: int = 4) -> str:
    if not sequence:
        return sequence
    arr = list(sequence)
    run = 1
    last = arr[0]
    for idx in range(1, len(arr)):
        if arr[idx] == last:
            run += 1
            if run > max_len:
                arr[idx] = random.choice([base for base in ALPHABET if base != arr[idx]])
                last = arr[idx]
                run = 1
        else:
            last = arr[idx]
            run = 1
    return "".join(arr)


def naturalize(sequence: str, background: str, gc_tol: float = 0.02, max_homopolymer: int = 4) -> str:
    return avoid_homopolymer(match_gc(sequence, gc_content(background), tol=gc_tol), max_homopolymer)


def replace_span(sequence: str, insert: str, start: int | None = None) -> tuple[str, int, int]:
    if len(insert) >= len(sequence):
        return insert[: len(sequence)], 0, len(sequence)
    if start is None:
        start = random.randint(0, len(sequence) - len(insert))
    end = start + len(insert)
    return sequence[:start] + insert + sequence[end:], start, end


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_dna(bits: str, mapping: dict[str, str] | None = None) -> str:
    mapping = mapping or random.choice(BIT_TO_DNA)
    if len(bits) % 2:
        bits += "0"
    return "".join(mapping[bits[idx : idx + 2]] for idx in range(0, len(bits), 2))


def checksum_bits(bits: str, width: int = 16) -> str:
    total = 0
    for idx in range(0, len(bits), width):
        total = (total + int(bits[idx : idx + width].ljust(width, "0"), 2)) % (2**width)
    return f"{total:0{width}b}"


def parity_ecc(bits: str, block: int = 8) -> str:
    out = []
    for idx in range(0, len(bits), block):
        chunk = bits[idx : idx + block]
        out.append(chunk + str(chunk.count("1") % 2))
    return "".join(out)


def repetition_ecc(bits: str, repeat: int = 3) -> str:
    return "".join(bit * repeat for bit in bits)


def stream_xor(bits: str, key: str = "carrier-key") -> str:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return "".join("1" if int(bit) ^ rng.randint(0, 1) else "0" for bit in bits)


def random_message(message_type: str | None = None, min_bytes: int = 8, max_bytes: int = 64) -> tuple[bytes, str]:
    message_type = message_type or random.choice(["random_bits", "ascii_text", "identifier", "timestamp"])
    max_bytes = max(min_bytes, max_bytes)
    if message_type == "random_bits":
        return bytes(random.getrandbits(8) for _ in range(random.randint(min_bytes, max_bytes))), message_type
    if message_type == "ascii_text":
        words = ["sample", "carrier", "watermark", "secret", "batch", "open-set", "codec", "message"]
        return " ".join(random.choice(words) for _ in range(random.randint(3, 12))).encode(), message_type
    if message_type == "identifier":
        return f"LAB-{random.randint(1000,9999)}-ID-{random.randint(100000,999999)}".encode(), message_type
    return f"TS-{random.randint(20200101,20261231)}-{random.randint(0,86400)}".encode(), "timestamp"


def preprocess_message(data: bytes, mode: str) -> tuple[str, dict[str, object]]:
    metadata: dict[str, object] = {"preprocess_mode": mode, "raw_bytes": len(data)}
    if mode == "plain":
        return bytes_to_bits(data), metadata
    if mode == "compressed":
        return bytes_to_bits(zlib.compress(data)), {**metadata, "compressed": True}
    if mode == "encrypted":
        return stream_xor(bytes_to_bits(data)), {**metadata, "encrypted_like": True}
    if mode == "compressed_encrypted":
        return stream_xor(bytes_to_bits(zlib.compress(data))), {**metadata, "compressed": True, "encrypted_like": True}
    return bytes_to_bits(data), metadata


def constrain_dna(sequence: str, target_background: str, gc_tol: float = 0.02, max_run: int = 3) -> str:
    return avoid_homopolymer(match_gc(sequence, gc_content(target_background), tol=gc_tol), max_run)


def packetize_bits(bits: str, payload_block: int = 96, index_bits: int = 16, checksum_width: int = 16) -> str:
    packets = []
    count = max(1, (len(bits) + payload_block - 1) // payload_block)
    for idx in range(count):
        block = bits[idx * payload_block : (idx + 1) * payload_block].ljust(payload_block, "0")
        index_bits_text = f"{idx:0{index_bits}b}"[-index_bits:]
        total_bits_text = f"{count:0{index_bits}b}"[-index_bits:]
        checksum = checksum_bits(index_bits_text + total_bits_text + block, width=checksum_width)
        packets.append(index_bits_text + total_bits_text + block + checksum)
    return "".join(packets)


def signature_like_bits(data: bytes, key: str = "v10-signature", bits: int = 128) -> str:
    digest = hashlib.sha256(key.encode() + data).hexdigest()
    return bin(int(digest, 16))[2:].zfill(256)[:bits]


CODON_TABLE = {
    "F": ["TTT", "TTC"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "Y": ["TAT", "TAC"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["TGT", "TGC"],
    "W": ["TGG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
}


class SemanticCarrierGenerator:
    codec_families = [
        "PlainCodec",
        "EncryptedCodec",
        "ECCCodec",
        "WatermarkCodec",
        "ProtocolCodec",
        "StegoCodec",
        "CodonStegoCodec",
        "KmerMatchedCodec",
        "DistributedCodec",
        "AdversarialMimicCodec",
    ]
    hard_negative_families = [
        "natural",
        "base_shuffle",
        "kmer_shuffle",
        "low_complexity",
        "microsatellite",
        "gc_patch",
        "at_patch",
        "periodic_decoy",
    ]
    rules = {
        "PlainCodec": ["plain_binary_map", "compressed_binary_map"],
        "EncryptedCodec": ["xor_stream", "compressed_xor_stream"],
        "ECCCodec": ["parity_ecc", "repetition3_ecc", "checksum_ecc"],
        "WatermarkCodec": ["sparse_id_watermark", "repeated_id_watermark"],
        "ProtocolCodec": ["header_payload_checksum", "versioned_packet", "dual_checksum_packet"],
        "StegoCodec": ["lsb_substitution", "gc_preserving_substitution"],
        "CodonStegoCodec": ["synonymous_bit", "synonymous_rank"],
        "KmerMatchedCodec": ["gc_matched_payload", "kmer_chunk_shuffle_payload"],
        "DistributedCodec": ["distributed_sparse_bits", "multi_segment_payload"],
        "AdversarialMimicCodec": ["background_mix", "weak_stego_mimic"],
    }

    def __init__(self, length: int = 1000, background_fasta: str | Path | None = None):
        self.length = int(length)
        self.bg_sampler = FastaBackgroundSampler(background_fasta, length=length)

    def make_background(self, split: str = "train", protocol: str = "standard") -> tuple[str, str]:
        return self.bg_sampler.sample(split=split, protocol=protocol)

    def negative(self, sequence: str, background_id: str, allow_hard: bool = True) -> CarrierResult:
        family = random.choice(self.hard_negative_families if allow_hard else ["natural"])
        out = sequence
        start = -1
        end = -1
        metadata: dict[str, object] = {"negative_type": family}
        if family == "base_shuffle":
            arr = list(sequence)
            random.shuffle(arr)
            out = "".join(arr)
        elif family == "kmer_shuffle":
            k = random.choice([2, 3, 4])
            chunks = [sequence[idx : idx + k] for idx in range(0, len(sequence), k)]
            random.shuffle(chunks)
            out = "".join(chunks)[: len(sequence)]
            metadata["k"] = k
        elif family == "low_complexity":
            length = min(random.randint(60, 200), max(1, len(sequence) // 2))
            start = random.randint(0, len(sequence) - length)
            end = start + length
            base = random.choice(ALPHABET)
            patch = "".join(base if random.random() < 0.86 else random.choice(ALPHABET) for _ in range(length))
            out = sequence[:start] + patch + sequence[end:]
        elif family == "microsatellite":
            motif = random.choice(["AT", "CA", "GT", "CAG", "GATA"])
            repeats = max(2, min(30, len(sequence) // max(1, len(motif) * 4)))
            patch = motif * random.randint(2, repeats)
            start = random.randint(0, len(sequence) - len(patch))
            end = start + len(patch)
            out = sequence[:start] + patch + sequence[end:]
        elif family in {"gc_patch", "at_patch"}:
            length = min(random.randint(80, 220), max(1, len(sequence) // 2))
            start = random.randint(0, len(sequence) - length)
            end = start + length
            probs = {"A": 0.12, "C": 0.38, "G": 0.38, "T": 0.12}
            if family == "at_patch":
                probs = {"A": 0.38, "C": 0.12, "G": 0.12, "T": 0.38}
            out = sequence[:start] + weighted_random_seq(length, probs) + sequence[end:]
        elif family == "periodic_decoy":
            length = min(random.randint(80, 180), max(1, len(sequence) // 2))
            start = random.randint(0, len(sequence) - length)
            end = start + length
            motif = random.choice(["AT", "CG", "TA"])
            patch = (motif * (length // len(motif) + 1))[:length]
            out = sequence[:start] + match_gc(patch, gc_content(sequence[start:end]), tol=0.08) + sequence[end:]

        carrier_type = "natural" if family == "natural" else "hard_natural_negative"
        return CarrierResult(out, 0, f"hard_negative:{family}", f"neg:{family}", carrier_type, "none", "none", "hard_negative", start, end, 0.0, background_id, 1, metadata)

    def _detectability(self, strength: float, codec: str) -> str:
        if codec in {"AdversarialMimicCodec", "CodonStegoCodec"}:
            return random.choice(["hard", "adversarial"])
        if strength > 0.72:
            return "easy"
        if strength > 0.42:
            return "medium"
        return "hard"

    def positive(
        self,
        sequence: str,
        codec_family: str,
        background_id: str,
        strength: float | None = None,
        codec_rule: str | None = None,
        detectability: str | None = None,
        message_type: str | None = None,
    ) -> CarrierResult:
        strength = float(strength if strength is not None else random.uniform(0.2, 1.0))
        codec_rule = codec_rule or random.choice(self.rules[codec_family])
        detectability = detectability or self._detectability(strength, codec_family)
        message, msg_type = random_message(message_type, max_bytes=int(24 + 64 * strength))
        return getattr(self, f"_codec_{codec_family}")(sequence, background_id, strength, codec_rule, detectability, message, msg_type)

    def _span_len(self, strength: float, lower: int, upper: int) -> int:
        return max(1, int(lower + strength * (upper - lower)))

    def _payload(self, message: bytes, msg_type: str, preprocess: str = "plain", ecc: str | None = None) -> tuple[str, dict[str, object]]:
        bits, metadata = preprocess_message(message, preprocess)
        if ecc == "parity":
            bits = parity_ecc(bits)
            metadata["ecc"] = "parity"
        elif ecc == "repetition3":
            bits = repetition_ecc(bits)
            metadata["ecc"] = "repetition3"
        elif ecc == "checksum":
            bits = bits + checksum_bits(bits)
            metadata["ecc"] = "checksum"
        dna = bits_to_dna(bits)
        metadata.update({"message_type": msg_type, "payload_bits": len(bits), "payload_dna_len": len(dna)})
        return dna, metadata

    def _wrap(self, sequence: str, codec: str, rule: str, msg_type: str, embedding: str, detectability: str, start: int, end: int, strength: float, background_id: str, metadata: dict[str, object]) -> CarrierResult:
        metadata = dict(metadata)
        metadata.update({"gc_after": gc_content(sequence), "entropy_after": sequence_entropy(sequence)})
        return CarrierResult(sequence, 1, codec, rule, "known_artificial_carrier", msg_type, embedding, detectability, start, end, strength, background_id, 0, metadata)

    def _codec_PlainCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, msg_type, "compressed" if "compressed" in rule else "plain")
        payload = payload[: self._span_len(strength, 80, 360)]
        if detect in {"hard", "adversarial"}:
            payload = naturalize(payload, sequence, 0.03)
        out, start, end = replace_span(sequence, payload)
        return self._wrap(out, "PlainCodec", rule, msg_type, "contiguous", detect, start, end, strength, bg, metadata)

    def _codec_EncryptedCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, "encrypted_payload", "compressed_encrypted" if "compressed" in rule else "encrypted")
        payload = naturalize(payload[: self._span_len(strength, 100, 420)], sequence, 0.05 if detect == "easy" else 0.015)
        out, start, end = replace_span(sequence, payload)
        return self._wrap(out, "EncryptedCodec", rule, "encrypted_payload", "contiguous", detect, start, end, strength, bg, metadata)

    def _codec_ECCCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        ecc = "parity" if "parity" in rule else ("repetition3" if "repetition3" in rule else "checksum")
        payload, metadata = self._payload(message, msg_type, "plain", ecc)
        payload = naturalize(payload[: self._span_len(strength, 100, 500)], sequence, 0.04)
        out, start, end = replace_span(sequence, payload)
        return self._wrap(out, "ECCCodec", rule, msg_type, "contiguous", detect, start, end, strength, bg, metadata)

    def _codec_WatermarkCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message[:16], "plain")
        arr = list(sequence)
        count = min(int(12 + strength * 80), len(sequence))
        positions = sorted(random.sample(range(len(sequence)), count))
        repeated = (bits * (len(positions) // max(1, len(bits)) + 1))[: len(positions)]
        pair = random.choice([("A", "G"), ("C", "T")])
        for position, bit in zip(positions, repeated):
            arr[position] = pair[int(bit)]
        if rule == "repeated_id_watermark":
            for position in positions[::4]:
                arr[position] = "G"
        metadata.update({"n_sites": len(positions), "positions_preview": positions[:20]})
        return self._wrap("".join(arr), "WatermarkCodec", rule, msg_type, "sparse_watermark", detect, min(positions), max(positions) + 1, strength, bg, metadata)

    def _codec_ProtocolCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, msg_type, "compressed" if strength > 0.5 else "plain", "checksum")
        header = weighted_random_seq(10 if rule != "versioned_packet" else 14)
        footer = weighted_random_seq(10)
        packet = header + bits_to_dna(f"{random.randint(1,15):04b}") + bits_to_dna(f"{len(payload):016b}") + payload
        if rule == "dual_checksum_packet":
            packet += bits_to_dna(checksum_bits(bytes_to_bits(message)[::-1]))
        packet = naturalize((packet + footer)[: self._span_len(strength, 120, 560)], sequence, 0.04)
        out, start, end = replace_span(sequence, packet)
        metadata.update({"packet_len": len(packet), "header_len": len(header)})
        return self._wrap(out, "ProtocolCodec", rule, msg_type, "protocol", detect, start, end, strength, bg, metadata)

    def _codec_StegoCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed_encrypted" if detect == "adversarial" else "plain")
        arr = list(sequence)
        count = min(len(bits), int(25 + strength * 160), len(sequence))
        positions = sorted(random.sample(range(len(sequence)), count))
        for position, bit in zip(positions, bits[:count]):
            old = arr[position]
            if old in "GC" and "gc_preserving" in rule:
                arr[position] = "G" if bit == "1" else "C"
            else:
                arr[position] = "T" if bit == "1" else "A"
        metadata.update({"n_sites": count, "positions_preview": positions[:20]})
        return self._wrap("".join(arr), "StegoCodec", rule, msg_type, "steganographic_substitution", detect, min(positions), max(positions) + 1, strength, bg, metadata)

    def _codec_CodonStegoCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "plain")
        count = min(int(35 + strength * 130), max(10, len(sequence) // 3 - 2))
        length = count * 3
        start = random.randint(0, max(0, len(sequence) - length))
        start -= start % 3
        if start + length > len(sequence):
            start = len(sequence) - length
        codons = []
        bit_index = 0
        choices = [aa for aa, codons_for_aa in CODON_TABLE.items() if len(codons_for_aa) > 1]
        for _ in range(count):
            synonyms = CODON_TABLE[random.choice(choices)]
            if rule == "synonymous_rank":
                two = bits[bit_index : bit_index + 2].ljust(2, "0")
                codons.append(synonyms[int(two, 2) % len(synonyms)])
                bit_index += 2
            else:
                codons.append(synonyms[int(bits[bit_index % len(bits)]) % len(synonyms)])
                bit_index += 1
        patch = naturalize("".join(codons), sequence[start : start + length], 0.03)
        metadata.update({"codon_count": count, "bits_used": bit_index})
        return self._wrap(sequence[:start] + patch + sequence[start + length :], "CodonStegoCodec", rule, msg_type, "codon_stego", detect, start, start + length, strength, bg, metadata)

    def _codec_KmerMatchedCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, msg_type, "compressed_encrypted", "checksum")
        length = min(len(payload), self._span_len(strength, 120, 460), len(sequence))
        start = random.randint(0, len(sequence) - length)
        end = start + length
        background_window = sequence[start:end]
        payload = naturalize(payload[:length], background_window, 0.01, 3)
        if "chunk_shuffle" in rule:
            k = random.choice([2, 3])
            chunks = [background_window[idx : idx + k] for idx in range(0, len(background_window), k)]
            random.shuffle(chunks)
            mixed = "".join(chunks)[:length]
            arr = list(payload)
            for idx in range(length):
                if random.random() < 0.35:
                    arr[idx] = mixed[idx]
            payload = "".join(arr)
            metadata["kmer_blend"] = True
        return self._wrap(sequence[:start] + payload + sequence[end:], "KmerMatchedCodec", rule, msg_type, "kmer_matched", detect, start, end, strength, bg, metadata)

    def _codec_DistributedCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, msg_type, "encrypted" if detect in {"hard", "adversarial"} else "plain")
        arr = list(sequence)
        positions: list[int] = []
        if rule == "distributed_sparse_bits":
            bits = bytes_to_bits(payload.encode())
            count = min(len(bits), int(20 + strength * 180), len(sequence))
            positions = sorted(random.sample(range(len(sequence)), count))
            for position, bit in zip(positions, bits[:count]):
                arr[position] = "G" if bit == "1" else "A"
        else:
            segments = random.randint(3, 9)
            chunk_len = max(8, int((80 + strength * 260) // segments))
            chunk_len = min(chunk_len, len(sequence))
            for _ in range(segments):
                position = random.randint(0, len(sequence) - chunk_len)
                chunk = naturalize(weighted_random_seq(chunk_len), sequence[position : position + chunk_len], 0.025)
                arr[position : position + chunk_len] = list(chunk)
                positions.extend(range(position, position + chunk_len))
            metadata["n_segments"] = segments
        metadata["positions_preview"] = positions[:20]
        return self._wrap("".join(arr), "DistributedCodec", rule, msg_type, "distributed", detect, min(positions), max(positions) + 1, strength, bg, metadata)

    def _codec_AdversarialMimicCodec(self, sequence: str, bg: str, strength: float, rule: str, detect: str, message: bytes, msg_type: str) -> CarrierResult:
        payload, metadata = self._payload(message, "encrypted_payload", "compressed_encrypted", "checksum")
        length = min(len(payload), self._span_len(strength, 120, 440), len(sequence))
        start = random.randint(0, len(sequence) - length)
        end = start + length
        background_window = sequence[start:end]
        payload = naturalize(payload[:length], background_window, 0.008, 3)
        arr = list(payload)
        keep = 0.55 if "weak" in rule else max(0.15, 0.65 - 0.45 * strength)
        for idx in range(length):
            if random.random() < keep:
                arr[idx] = background_window[idx]
        metadata.update({"keep_rate": keep, "adversarial_mimic": True})
        return self._wrap(sequence[:start] + "".join(arr) + sequence[end:], "AdversarialMimicCodec", rule, "encrypted_payload", "adversarial_mimic", "adversarial", start, end, strength, bg, metadata)


class LiteratureInspiredCarrierGenerator(SemanticCarrierGenerator):
    literature_codecs = [
        "ConstrainedStorageCodec",
        "IndexedPayloadCodec",
        "FountainLikeCodec",
        "SignatureWatermarkCodec",
        "TraceabilityTagCodec",
        "DNACryptoCodec",
        "AdaptiveStegoCodec",
        "ErrorRobustCodec",
    ]
    literature_rules = {
        "ConstrainedStorageCodec": ["gc_balanced", "homopolymer_limited", "balanced_packet"],
        "IndexedPayloadCodec": ["index_payload_checksum", "dual_index_packet"],
        "FountainLikeCodec": ["xor_droplet", "seeded_droplet"],
        "SignatureWatermarkCodec": ["signature_tag", "signature_payload_tag"],
        "TraceabilityTagCodec": ["lab_batch_id", "distributed_trace_id"],
        "DNACryptoCodec": ["stream_cipher_like", "compressed_stream_cipher_like"],
        "AdaptiveStegoCodec": ["background_adaptive_sparse", "kmer_adaptive_patch"],
        "ErrorRobustCodec": ["repetition_ecc_packet", "parity_ecc_packet"],
    }

    def __init__(self, length: int = 1000, background_fasta: str | Path | None = None, include_original: bool = True):
        super().__init__(length=length, background_fasta=background_fasta)
        if include_original:
            self.codec_families = self.codec_families + self.literature_codecs
            self.rules = {**self.rules, **self.literature_rules}
        else:
            self.codec_families = list(self.literature_codecs)
            self.rules = dict(self.literature_rules)

    def positive(self, sequence: str, codec_family: str, background_id: str, strength: float | None = None, codec_rule: str | None = None, detectability: str | None = None, message_type: str | None = None) -> CarrierResult:
        if codec_family not in self.literature_codecs:
            return super().positive(sequence, codec_family, background_id, strength, codec_rule, detectability, message_type)
        strength = float(strength if strength is not None else random.uniform(0.2, 1.0))
        codec_rule = codec_rule or random.choice(self.rules[codec_family])
        detectability = detectability or self._detectability(strength, codec_family)
        message, msg_type = random_message(message_type=message_type, min_bytes=12, max_bytes=int(32 + 96 * strength))
        return getattr(self, f"_lit_{codec_family}")(sequence, background_id, strength, codec_rule, detectability, message, msg_type)

    def _lit_ConstrainedStorageCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed")
        if rule == "balanced_packet":
            bits = packetize_bits(bits)
        dna = bits_to_dna(bits)
        length = min(len(dna), int(120 + strength * 520), len(sequence))
        patch = constrain_dna(dna[:length], sequence, gc_tol=0.012 if detectability != "easy" else 0.035)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"literature_inspired": True, "constraint": rule})
        return self._wrap(out, "ConstrainedStorageCodec", rule, msg_type, "constrained_storage", detectability, start, end, strength, bg, metadata)

    def _lit_IndexedPayloadCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed")
        block = 80 if rule == "dual_index_packet" else 112
        packet = packetize_bits(bits, payload_block=block)
        if rule == "dual_index_packet":
            packet += packetize_bits(bits[::-1], payload_block=block)
        dna = bits_to_dna(packet)
        length = min(len(dna), int(160 + strength * 620), len(sequence))
        patch = constrain_dna(dna[:length], sequence)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"indexed": True, "packet_bits": len(packet)})
        return self._wrap(out, "IndexedPayloadCodec", rule, msg_type, "indexed_payload", detectability, start, end, strength, bg, metadata)

    def _lit_FountainLikeCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed")
        rng = random.Random(int(hashlib.md5(message).hexdigest()[:8], 16))
        chunk_size = 64
        chunks = [bits[idx : idx + chunk_size].ljust(chunk_size, "0") for idx in range(0, len(bits), chunk_size)] or ["0" * chunk_size]
        droplets = []
        droplet_count = int(6 + strength * 28)
        for _ in range(droplet_count):
            seed = rng.randint(0, 2**16 - 1)
            degree = rng.randint(1, min(4, len(chunks)))
            chosen = rng.sample(chunks, degree)
            acc = 0
            for chunk in chosen:
                acc ^= int(chunk, 2)
            droplets.append(f"{seed:016b}" + f"{degree:04b}" + f"{acc:0{chunk_size}b}" + checksum_bits(f"{seed:016b}{acc:0{chunk_size}b}"))
        dna = bits_to_dna("".join(droplets))
        length = min(len(dna), int(180 + strength * 640), len(sequence))
        patch = constrain_dna(dna[:length], sequence)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"fountain_like": True, "n_droplets": droplet_count})
        return self._wrap(out, "FountainLikeCodec", rule, msg_type, "fountain_like", detectability, start, end, strength, bg, metadata)

    def _lit_SignatureWatermarkCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        signature = signature_like_bits(message)
        payload, _ = preprocess_message(message, "compressed") if rule == "signature_payload_tag" else ("", {})
        dna = bits_to_dna(signature + payload[: int(64 + strength * 256)])
        arr = list(sequence)
        count = min(len(dna), int(24 + strength * 160), len(sequence))
        positions = sorted(random.sample(range(len(sequence)), count))
        for position, base in zip(positions, dna[:count]):
            arr[position] = base
        metadata = {"signature_like": True, "n_sites": count, "positions_preview": positions[:20]}
        return self._wrap("".join(arr), "SignatureWatermarkCodec", rule, msg_type, "signature_watermark", detectability, min(positions), max(positions) + 1, strength, bg, metadata)

    def _lit_TraceabilityTagCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        trace = f"LAB{random.randint(100,999)}-BATCH{random.randint(1000,9999)}-S{random.randint(100000,999999)}".encode()
        bits, metadata = preprocess_message(trace, "plain")
        bits = bits + checksum_bits(bits)
        dna = bits_to_dna(bits)
        if rule == "distributed_trace_id":
            arr = list(sequence)
            count = min(len(dna), int(32 + strength * 120), len(sequence))
            positions = sorted(random.sample(range(len(sequence)), count))
            for position, base in zip(positions, dna[:count]):
                arr[position] = base
            out = "".join(arr)
            start = min(positions)
            end = max(positions) + 1
            embedding = "distributed_trace"
        else:
            patch = constrain_dna(dna, sequence)
            out, start, end = replace_span(sequence, patch)
            embedding = "traceability_tag"
        metadata.update({"traceability_tag": True, "trace_text": trace.decode(errors="ignore")})
        return self._wrap(out, "TraceabilityTagCodec", rule, "identifier", embedding, detectability, start, end, strength, bg, metadata)

    def _lit_DNACryptoCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        mode = "compressed_encrypted" if rule == "compressed_stream_cipher_like" else "encrypted"
        bits, metadata = preprocess_message(message, mode)
        nonce = f"{random.getrandbits(32):032b}"
        auth = signature_like_bits(message, key="crypto-auth", bits=64)
        dna = bits_to_dna(nonce + bits + auth)
        length = min(len(dna), int(150 + strength * 620), len(sequence))
        patch = constrain_dna(dna[:length], sequence, gc_tol=0.015)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"crypto_like": True, "nonce_bits": 32, "auth_bits": 64})
        return self._wrap(out, "DNACryptoCodec", rule, "encrypted_payload", "dna_crypto", detectability, start, end, strength, bg, metadata)

    def _lit_AdaptiveStegoCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed_encrypted")
        arr = list(sequence)
        count = min(len(bits), int(30 + strength * 220), len(sequence))
        positions = sorted(random.sample(range(len(sequence)), count))
        for position, bit in zip(positions, bits[:count]):
            old = arr[position]
            if rule == "background_adaptive_sparse":
                arr[position] = ("G" if bit == "1" else "C") if old in "GC" else ("T" if bit == "1" else "A")
            else:
                arr[position] = old if random.random() < 0.45 else ("G" if bit == "1" else "A")
        metadata.update({"adaptive_stego": True, "n_sites": count, "positions_preview": positions[:20]})
        return self._wrap("".join(arr), "AdaptiveStegoCodec", rule, "encrypted_payload", "adaptive_stego", "adversarial" if detectability == "hard" else detectability, min(positions), max(positions) + 1, strength, bg, metadata)

    def _lit_ErrorRobustCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed")
        bits = repetition_ecc(bits, 3) if rule == "repetition_ecc_packet" else parity_ecc(bits)
        bits = packetize_bits(bits, payload_block=80)
        dna = bits_to_dna(bits)
        length = min(len(dna), int(180 + strength * 680), len(sequence))
        patch = constrain_dna(dna[:length], sequence)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"error_robust": True, "ecc_rule": rule})
        return self._wrap(out, "ErrorRobustCodec", rule, msg_type, "error_robust_storage", detectability, start, end, strength, bg, metadata)


class RealisticCarrierGenerator(LiteratureInspiredCarrierGenerator):
    realistic_codecs = [
        "RealisticStorageCodec",
        "RealisticSparseWatermarkCodec",
        "RealisticTraceabilityCodec",
        "RealisticAdaptiveStegoCodec",
    ]
    realistic_rules = {
        "RealisticStorageCodec": ["primer_index_payload_ecc", "multi_oligo_packet"],
        "RealisticSparseWatermarkCodec": ["distributed_signature_sites", "gc_preserving_signature_sites"],
        "RealisticTraceabilityCodec": ["issuer_batch_signature", "self_documenting_tag"],
        "RealisticAdaptiveStegoCodec": ["local_gc_matched_sparse", "local_kmer_mimic_patch"],
    }

    def __init__(self, length: int = 1000, background_fasta: str | Path | None = None):
        super().__init__(length=length, background_fasta=background_fasta, include_original=True)
        self.codec_families = self.codec_families + self.realistic_codecs
        self.rules = {**self.rules, **self.realistic_rules}

    def positive(self, sequence: str, codec_family: str, background_id: str, strength: float | None = None, codec_rule: str | None = None, detectability: str | None = None, message_type: str | None = None) -> CarrierResult:
        if codec_family not in self.realistic_codecs:
            return super().positive(sequence, codec_family, background_id, strength, codec_rule, detectability, message_type)
        strength = float(strength if strength is not None else random.uniform(0.2, 1.0))
        codec_rule = codec_rule or random.choice(self.rules[codec_family])
        detectability = detectability or self._detectability(strength, codec_family)
        message, msg_type = random_message(message_type=message_type, min_bytes=16, max_bytes=int(48 + 128 * strength))
        return getattr(self, f"_realistic_{codec_family}")(sequence, background_id, strength, codec_rule, detectability, message, msg_type)

    def _realistic_RealisticStorageCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed")
        packet = packetize_bits(bits, payload_block=96)
        primer_left = bits_to_dna(signature_like_bits(message, "left", 24))
        primer_right = bits_to_dna(signature_like_bits(message, "right", 24))
        dna = primer_left + bits_to_dna(packet) + primer_right
        if rule == "multi_oligo_packet":
            dna = "".join(dna[idx : idx + 120] + constrain_dna(weighted_random_seq(6), sequence, 0.05) for idx in range(0, len(dna), 120))
        length = min(len(dna), int(180 + strength * 720), len(sequence))
        patch = constrain_dna(dna[:length], sequence, 0.012 if detectability != "easy" else 0.03, 3)
        out, start, end = replace_span(sequence, patch)
        metadata.update({"v11_realistic": True, "packet_bits": len(packet), "patch_len": len(patch)})
        return self._wrap(out, "RealisticStorageCodec", rule, msg_type, "realistic_storage", detectability, start, end, strength, bg, metadata)

    def _realistic_RealisticSparseWatermarkCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        dna = bits_to_dna(signature_like_bits(message, "v11-watermark", 192))
        arr = list(sequence)
        count = min(len(dna), int(32 + strength * 180), len(sequence))
        positions = sorted(random.sample(range(len(sequence)), count))
        for position, base in zip(positions, dna[:count]):
            old = arr[position]
            if rule == "gc_preserving_signature_sites":
                arr[position] = ("G" if base in "GT" else "C") if old in "GC" else ("T" if base in "GT" else "A")
            else:
                arr[position] = base
        metadata = {"v11_realistic": True, "signature_sites": count, "positions_preview": positions[:25]}
        return self._wrap("".join(arr), "RealisticSparseWatermarkCodec", rule, "identifier", "sparse_watermark_sites", detectability, min(positions), max(positions) + 1, strength, bg, metadata)

    def _realistic_RealisticTraceabilityCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        issuer = f"ISSUER-{random.randint(100,999)}"
        batch = f"BATCH-{random.randint(1000,9999)}"
        cert = f"{issuer}|{batch}|LEN{len(message)}".encode()
        bits = bytes_to_bits(cert) + signature_like_bits(cert + message, "trace", 128) + checksum_bits(bytes_to_bits(cert), width=16)
        if rule == "self_documenting_tag":
            bits += bytes_to_bits(b"|DOC|V11|")
        patch = constrain_dna(bits_to_dna(bits)[: int(100 + strength * 420)], sequence, 0.02, 3)
        out, start, end = replace_span(sequence, patch)
        return self._wrap(out, "RealisticTraceabilityCodec", rule, "identifier", "traceability_certificate", detectability, start, end, strength, bg, {"issuer": issuer, "batch": batch, "signature_like": True})

    def _realistic_RealisticAdaptiveStegoCodec(self, sequence: str, bg: str, strength: float, rule: str, detectability: str, message: bytes, msg_type: str) -> CarrierResult:
        bits, metadata = preprocess_message(message, "compressed_encrypted")
        arr = list(sequence)
        count = min(len(bits), int(40 + strength * 260), max(0, len(sequence) - 4))
        candidates = list(range(2, len(sequence) - 2))
        random.shuffle(candidates)
        positions = sorted(candidates[:count])
        for position, bit in zip(positions, bits[:count]):
            local = sequence[max(0, position - 2) : min(len(sequence), position + 3)]
            old = arr[position]
            if rule == "local_gc_matched_sparse":
                arr[position] = ("G" if bit == "1" else "C") if gc_content(local) >= 0.5 else ("T" if bit == "1" else "A")
            else:
                arr[position] = old if random.random() < 0.55 else ("G" if bit == "1" else "A")
        metadata.update({"v11_realistic": True, "adaptive_sites": len(positions), "positions_preview": positions[:25]})
        return self._wrap("".join(arr), "RealisticAdaptiveStegoCodec", rule, "encrypted_payload", "realistic_adaptive_stego", "adversarial" if detectability in {"hard", "adversarial"} else detectability, min(positions), max(positions) + 1, strength, bg, metadata)


def make_generator(kind: str, length: int, background_fasta: str | Path | None = None) -> SemanticCarrierGenerator:
    if kind == "base":
        return SemanticCarrierGenerator(length=length, background_fasta=background_fasta)
    if kind == "literature":
        return LiteratureInspiredCarrierGenerator(length=length, background_fasta=background_fasta)
    if kind == "realistic":
        return RealisticCarrierGenerator(length=length, background_fasta=background_fasta)
    raise ValueError(f"Unknown carrier generator kind: {kind}")


def open_target(y: int, is_unknown: int) -> str:
    return "natural" if y == 0 else ("unknown_anomaly" if is_unknown else "known_special")


def _choose_detectability(protocol: str, split: str, train_detectability: list[str], test_detectability: list[str]) -> str | None:
    if protocol == "detectability_shift":
        return random.choice(test_detectability if split == "test" else train_detectability)
    return None


def _result_to_row(result: CarrierResult, split: str, is_unknown: int) -> dict[str, object]:
    carrier_type = result.carrier_type
    if result.y == 1:
        carrier_type = "unknown_artificial_carrier" if is_unknown else "known_artificial_carrier"
    return {
        "split": split,
        "sequence": result.sequence,
        "y": result.y,
        "family": result.codec_family,
        "rule_id": result.codec_rule,
        "start": result.start,
        "end": result.end,
        "strength": result.strength,
        "background_id": result.background_id,
        "naturalness_label": result.naturalness_label,
        "is_unknown": int(is_unknown),
        "open_target": open_target(result.y, is_unknown),
        "codec_family": result.codec_family,
        "codec_rule": result.codec_rule,
        "carrier_type": carrier_type,
        "message_type": result.message_type,
        "embedding_type": result.embedding_type,
        "detectability": result.detectability,
        "payload_start": result.start,
        "payload_end": result.end,
        "metadata": json.dumps(result.metadata, ensure_ascii=False),
    }


def generate_split(
    generator: SemanticCarrierGenerator,
    split: str,
    n: int,
    protocol: str,
    train_codecs: list[str],
    holdout_codecs: list[str],
    negative_rate: float,
    include_unknown: bool,
    train_detectability: list[str],
    test_detectability: list[str],
    allow_hard_negatives: bool = True,
) -> list[dict[str, object]]:
    rows = []
    for sample_index in range(n):
        background, background_id = generator.make_background(split=split, protocol=protocol)
        if random.random() < negative_rate:
            result = generator.negative(background, background_id, allow_hard=allow_hard_negatives)
            is_unknown = 0
        else:
            use_unknown = include_unknown and bool(holdout_codecs) and random.random() < 0.45
            codec = random.choice(holdout_codecs if use_unknown else train_codecs)
            is_unknown = int(use_unknown)
            detectability = _choose_detectability(protocol, split, train_detectability, test_detectability)
            if protocol == "detectability_shift" and split == "test":
                is_unknown = 1
            strength = random.uniform(0.2, 1.0)
            if detectability == "easy":
                strength = random.uniform(0.72, 1.0)
            elif detectability == "medium":
                strength = random.uniform(0.42, 0.72)
            elif detectability == "hard":
                strength = random.uniform(0.20, 0.48)
            elif detectability == "adversarial":
                strength = random.uniform(0.20, 0.75)
            result = generator.positive(background, codec, background_id, strength=strength, detectability=detectability)
        row = _result_to_row(result, split, is_unknown)
        row["sample_id"] = f"{split}_{sample_index:08d}"
        rows.append(row)
    return rows


def build_carrier_benchmark(
    out: str | Path,
    *,
    generator_kind: str = "realistic",
    protocol: str = "standard",
    length: int = 1000,
    n_train: int = 6000,
    n_val: int = 1500,
    n_test: int = 2500,
    holdout_codecs: list[str] | None = None,
    calibration_codecs: list[str] | None = None,
    negative_rate: float = 0.35,
    train_detectability: list[str] | None = None,
    test_detectability: list[str] | None = None,
    seed: int = 42,
    allow_hard_negatives: bool = True,
    background_fasta: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    set_seed(seed)
    generator = make_generator(generator_kind, length=length, background_fasta=background_fasta)
    holdout_codecs = holdout_codecs if holdout_codecs is not None else ["CodonStegoCodec", "AdversarialMimicCodec"]
    if calibration_codecs is None:
        calibration_codecs = [
            codec
            for codec in ["StegoCodec"]
            if codec in generator.codec_families and codec not in set(holdout_codecs)
        ]
    calibration_codecs = [
        codec
        for codec in calibration_codecs
        if codec in generator.codec_families and codec not in set(holdout_codecs)
    ]
    train_detectability = train_detectability or ["easy", "medium"]
    test_detectability = test_detectability or ["hard", "adversarial"]
    reserved_codecs = set(holdout_codecs) | set(calibration_codecs)
    train_codecs = [codec for codec in generator.codec_families if codec not in reserved_codecs]
    if not train_codecs:
        raise ValueError("No training codecs left after holdout.")

    rows = []
    rows += generate_split(generator, "train", n_train, protocol, train_codecs, holdout_codecs, negative_rate, False, train_detectability, test_detectability, allow_hard_negatives)
    rows += generate_split(
        generator,
        "val",
        n_val,
        protocol,
        train_codecs,
        calibration_codecs,
        negative_rate,
        bool(calibration_codecs),
        train_detectability,
        test_detectability,
        allow_hard_negatives,
    )
    rows += generate_split(generator, "test", n_test, protocol, train_codecs, holdout_codecs, negative_rate, True, train_detectability, test_detectability, allow_hard_negatives)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(out_path, rows)

    manifest: dict[str, object] = {
        "name": "genome_mining V12 carrier controls",
        "source": "Migrated from dna_semantic_carrier_v12 V8-V12 codec lineage",
        "task": "Open-set detection of artificially embedded semantic information carriers",
        "generator_kind": generator_kind,
        "protocol": protocol,
        "length": length,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "all_codecs": generator.codec_families,
        "train_codecs": train_codecs,
        "calibration_codecs": calibration_codecs,
        "holdout_codecs": holdout_codecs,
        "negative_rate": negative_rate,
        "train_detectability": train_detectability,
        "test_detectability": test_detectability,
        "allow_hard_negatives": allow_hard_negatives,
        "background_fasta": str(background_fasta) if background_fasta else None,
        "seed": seed,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, manifest


def _write_rows_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)
