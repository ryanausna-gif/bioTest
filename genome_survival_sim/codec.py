from __future__ import annotations

import hashlib
import hmac
import math
import struct
from dataclasses import dataclass
from typing import Iterable


DNA_ALPHABET = "ACGT"
DNA_TO_BITS = {base: index for index, base in enumerate(DNA_ALPHABET)}
MAGIC = b"DSIM1"
HEADER = struct.Struct(">5s8sHHHHL12s")
CHUNK_HEADER = struct.Struct(">HHHL")


@dataclass(frozen=True)
class PayloadCodecConfig:
    block_size_bytes: int = 128
    copies_per_block: int = 2
    sync_marker_bp: int = 32
    auth_tag_bytes: int = 16
    sync_mismatches: int = 0
    secret: str = "replace-this-project-secret"
    encryption_mode: str = "hmac_stream"
    ecc_mode: str = "none"
    ecc_symbols: int = 16
    erasure_mode: str = "none"
    erasure_group_size: int = 8
    erasure_repair_blocks: int = 4
    resync_mode: str = "none"
    resync_chunk_bytes: int = 32
    resync_marker_bp: int = 24
    resync_mismatches: int = 0


@dataclass(frozen=True)
class EncodedFragment:
    block_id: int
    total_blocks: int
    copy_id: int
    dna: str

    @property
    def length(self) -> int:
        return len(self.dna)


@dataclass(frozen=True)
class DecodedBlock:
    block_id: int
    total_blocks: int
    payload_length: int
    plain: bytes


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    payload: bytes
    total_blocks: int
    recovered_blocks: int
    marker_seen: int
    valid_fragments: int
    reason: str


def bytes_to_dna(data: bytes) -> str:
    bases: list[str] = []
    for value in data:
        bases.append(DNA_ALPHABET[(value >> 6) & 0b11])
        bases.append(DNA_ALPHABET[(value >> 4) & 0b11])
        bases.append(DNA_ALPHABET[(value >> 2) & 0b11])
        bases.append(DNA_ALPHABET[value & 0b11])
    return "".join(bases)


def dna_to_bytes(dna: str) -> bytes:
    if len(dna) % 4 != 0:
        raise ValueError("DNA byte encoding length must be divisible by 4.")
    output = bytearray()
    for index in range(0, len(dna), 4):
        value = 0
        for base in dna[index : index + 4]:
            value = (value << 2) | DNA_TO_BITS[base]
        output.append(value)
    return bytes(output)


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


class PayloadCodec:
    """Authenticated byte payload codec with DNA serialization.

    Supported encryption modes:
    - hmac_stream: dependency-free simulation mode.
    - chacha20_poly1305: real AEAD mode via the optional cryptography package.

    Supported ECC modes:
    - none: no block-level error correction.
    - reed_solomon: byte-level Reed-Solomon parity via the optional reedsolo package.

    Supported erasure modes:
    - none: no cross-block erasure recovery.
    - xor_parity: add one parity block per group, recovering one lost data block per group.
    - fountain: add multiple GF(256) repair blocks per group, recovering multiple lost blocks when rank permits.

    Supported resynchronization modes:
    - none: serialize the full packet as one continuous byte-to-DNA body.
    - chunked: split the packet into marker-delimited chunks so later chunks can resync after indels.
    """

    def __init__(self, config: PayloadCodecConfig) -> None:
        self.config = config
        root_key = hashlib.sha256(config.secret.encode("utf-8")).digest()
        self._enc_key = hmac.new(root_key, b"enc", hashlib.sha256).digest()
        self._auth_key = hmac.new(root_key, b"auth", hashlib.sha256).digest()
        self._marker_key = hmac.new(root_key, b"marker", hashlib.sha256).digest()
        self._encryption_mode = normalize_encryption_mode(config.encryption_mode)
        self._ecc_mode = normalize_ecc_mode(config.ecc_mode)
        self._erasure_mode = normalize_erasure_mode(config.erasure_mode)
        self._resync_mode = normalize_resync_mode(config.resync_mode)

    def payload_id(self, payload: bytes) -> bytes:
        return hmac.new(self._auth_key, b"payload-id" + payload, hashlib.sha256).digest()[:8]

    def sync_marker(self) -> str:
        material = b""
        counter = 0
        while len(material) * 4 < self.config.sync_marker_bp:
            material += hmac.new(
                self._marker_key,
                b"sync-v1" + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            counter += 1
        return bytes_to_dna(material)[: self.config.sync_marker_bp]

    def encode(self, payload: bytes) -> list[EncodedFragment]:
        if self.config.block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be positive.")
        if self.config.block_size_bytes > 65_535:
            raise ValueError("block_size_bytes cannot exceed 65535 with the current block header.")
        if self.config.copies_per_block <= 0:
            raise ValueError("copies_per_block must be positive.")
        if self.config.copies_per_block > 65_535:
            raise ValueError("copies_per_block cannot exceed 65535 with the current block header.")
        if self._resync_mode == "chunked":
            if self.config.resync_chunk_bytes <= 0:
                raise ValueError("resync_chunk_bytes must be positive for chunked resync.")
            if self.config.resync_chunk_bytes > 65_535:
                raise ValueError("resync_chunk_bytes cannot exceed 65535 with the current chunk header.")
            if self.config.resync_marker_bp <= 0:
                raise ValueError("resync_marker_bp must be positive for chunked resync.")
            if self.config.resync_mismatches < 0:
                raise ValueError("resync_mismatches cannot be negative.")

        payload_id = self.payload_id(payload)
        marker = self.sync_marker()
        payload_length = len(payload)
        if payload_length > 4_294_967_295:
            raise ValueError("payload length cannot exceed 2^32-1 bytes with the current block header.")
        total_blocks = max(1, math.ceil(payload_length / self.config.block_size_bytes))
        self._validate_block_count(total_blocks)
        fragments: list[EncodedFragment] = []
        plain_blocks: list[bytes] = []

        for block_id in range(total_blocks):
            start = block_id * self.config.block_size_bytes
            plain = payload[start : start + self.config.block_size_bytes]
            plain_blocks.append(plain)
            for copy_id in range(self.config.copies_per_block):
                fragments.append(
                    self._encode_fragment(
                        marker=marker,
                        payload_id=payload_id,
                        block_id=block_id,
                        total_blocks=total_blocks,
                        copy_id=copy_id,
                        payload_length=payload_length,
                        plain=plain,
                    )
                )

        if self._erasure_mode in {"xor_parity", "fountain"}:
            for repair_block_id, repair_plain in self._erasure_repair_blocks(plain_blocks, total_blocks):
                for copy_id in range(self.config.copies_per_block):
                    fragments.append(
                        self._encode_fragment(
                            marker=marker,
                            payload_id=payload_id,
                            block_id=repair_block_id,
                            total_blocks=total_blocks,
                            copy_id=copy_id,
                            payload_length=payload_length,
                            plain=repair_plain,
                        )
                    )
        return fragments

    def decode_fragments(self, fragment_dna: Iterable[str], expected_payload: bytes | None = None) -> DecodeResult:
        marker_seen = 0
        valid_fragments = 0
        blocks: dict[int, DecodedBlock] = {}
        repair_blocks: dict[tuple[int, int], DecodedBlock] = {}
        total_blocks: int | None = None
        payload_length: int | None = None

        for dna in fragment_dna:
            marker_found, block = self._decode_one(dna)
            if marker_found:
                marker_seen += 1
            if block is None:
                continue
            valid_fragments += 1
            total_blocks = block.total_blocks
            payload_length = block.payload_length
            if 0 <= block.block_id < block.total_blocks:
                blocks.setdefault(block.block_id, block)
            elif self._erasure_mode in {"xor_parity", "fountain"}:
                repair_key = self._repair_key(block.block_id, block.total_blocks)
                if repair_key is not None:
                    repair_blocks.setdefault(repair_key, block)

        if total_blocks is None:
            reason = "sync_marker_destroyed" if not marker_seen else "auth_failed"
            return DecodeResult(False, b"", 0, 0, marker_seen, valid_fragments, reason)

        if payload_length is None:
            payload_length = 0
        if self._erasure_mode == "xor_parity":
            self._recover_xor_parity_blocks(blocks, repair_blocks, total_blocks, payload_length)
        elif self._erasure_mode == "fountain":
            self._recover_fountain_blocks(blocks, repair_blocks, total_blocks, payload_length)
        recovered_blocks = len(blocks)
        if recovered_blocks < total_blocks:
            return DecodeResult(
                False,
                b"",
                total_blocks,
                recovered_blocks,
                marker_seen,
                valid_fragments,
                "block_missing_or_mutated",
            )

        payload = b"".join(blocks[index].plain for index in range(total_blocks))[:payload_length]
        if expected_payload is not None and payload != expected_payload:
            return DecodeResult(
                False,
                payload,
                total_blocks,
                recovered_blocks,
                marker_seen,
                valid_fragments,
                "payload_mismatch",
            )
        return DecodeResult(True, payload, total_blocks, recovered_blocks, marker_seen, valid_fragments, "success")

    def _decode_one(self, dna: str) -> tuple[bool, DecodedBlock | None]:
        marker_pos = self._find_marker(dna)
        if marker_pos < 0:
            return False, None

        body = dna[marker_pos + self.config.sync_marker_bp :]
        try:
            raw, erase_positions = self._deserialize_body(body)
        except (KeyError, ValueError):
            return True, None

        try:
            packet = self._ecc_decode(raw, erase_positions)
        except ValueError:
            return True, None

        if len(packet) < HEADER.size + self.config.auth_tag_bytes:
            return True, None

        header = packet[: HEADER.size]
        try:
            magic, _payload_id, block_id, total_blocks, _copy_id, plain_len, payload_length, nonce = HEADER.unpack(header)
        except struct.error:
            return True, None
        if magic != MAGIC:
            return True, None

        encrypted_start = HEADER.size
        encrypted_len = self._encrypted_length(plain_len)
        encrypted_end = encrypted_start + encrypted_len
        if plain_len > self.config.block_size_bytes or encrypted_end > len(packet):
            return True, None

        encrypted = packet[encrypted_start:encrypted_end]
        try:
            plain = self._decrypt_block(header=header, encrypted=encrypted, nonce=nonce, plain_len=plain_len)
        except ValueError:
            return True, None

        return True, DecodedBlock(
            block_id=block_id,
            total_blocks=total_blocks,
            payload_length=payload_length,
            plain=plain,
        )

    def _find_marker(self, dna: str) -> int:
        marker = self.sync_marker()
        window = len(marker)
        max_mismatches = self.config.sync_mismatches
        for index in range(0, max(0, len(dna) - window) + 1):
            candidate = dna[index : index + window]
            if len(candidate) != window:
                continue
            if _hamming(candidate, marker) <= max_mismatches:
                return index
        return -1

    def marker_for_expected_payload(self, payload: bytes) -> str:
        return self.sync_marker()

    def with_expected_payload_marker(self, payload: bytes) -> "ExpectedMarkerPayloadCodec":
        return ExpectedMarkerPayloadCodec(self.config, self.marker_for_expected_payload(payload))

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        output = bytearray()
        counter = 0
        while len(output) < length:
            output.extend(
                hmac.new(
                    self._enc_key,
                    b"stream" + nonce + counter.to_bytes(4, "big"),
                    hashlib.sha256,
                ).digest()
            )
            counter += 1
        return bytes(output[:length])

    def _encrypt_block(self, header: bytes, plain: bytes, nonce: bytes) -> bytes:
        if self._encryption_mode == "hmac_stream":
            ciphertext = _xor(plain, self._keystream(nonce, len(plain)))
            tag = hmac.new(self._auth_key, header + ciphertext, hashlib.sha256).digest()[
                : self.config.auth_tag_bytes
            ]
            return ciphertext + tag
        if self._encryption_mode == "chacha20_poly1305":
            if self.config.auth_tag_bytes != 16:
                raise ValueError("ChaCha20-Poly1305 uses a fixed 16-byte authentication tag.")
            return _chacha20_poly1305_encrypt(self._enc_key, nonce, plain, header)
        raise ValueError(f"Unsupported encryption mode: {self.config.encryption_mode}")

    def _decrypt_block(self, header: bytes, encrypted: bytes, nonce: bytes, plain_len: int) -> bytes:
        if self._encryption_mode == "hmac_stream":
            if len(encrypted) != plain_len + self.config.auth_tag_bytes:
                raise ValueError("Invalid hmac_stream encrypted block length.")
            ciphertext = encrypted[:plain_len]
            tag = encrypted[plain_len:]
            expected_tag = hmac.new(self._auth_key, header + ciphertext, hashlib.sha256).digest()[
                : self.config.auth_tag_bytes
            ]
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed.")
            return _xor(ciphertext, self._keystream(nonce, plain_len))
        if self._encryption_mode == "chacha20_poly1305":
            if len(encrypted) != plain_len + 16:
                raise ValueError("Invalid ChaCha20-Poly1305 encrypted block length.")
            return _chacha20_poly1305_decrypt(self._enc_key, nonce, encrypted, header)
        raise ValueError(f"Unsupported encryption mode: {self.config.encryption_mode}")

    def _encrypted_length(self, plain_len: int) -> int:
        if self._encryption_mode == "hmac_stream":
            return plain_len + self.config.auth_tag_bytes
        if self._encryption_mode == "chacha20_poly1305":
            return plain_len + 16
        raise ValueError(f"Unsupported encryption mode: {self.config.encryption_mode}")

    def _encode_fragment(
        self,
        marker: str,
        payload_id: bytes,
        block_id: int,
        total_blocks: int,
        copy_id: int,
        payload_length: int,
        plain: bytes,
    ) -> EncodedFragment:
        nonce = hmac.new(
            self._enc_key,
            payload_id
            + block_id.to_bytes(2, "big")
            + copy_id.to_bytes(2, "big"),
            hashlib.sha256,
        ).digest()[:12]
        header = HEADER.pack(
            MAGIC,
            payload_id,
            block_id,
            total_blocks,
            copy_id,
            len(plain),
            payload_length,
            nonce,
        )
        encrypted = self._encrypt_block(header=header, plain=plain, nonce=nonce)
        packet = self._ecc_encode(header + encrypted)
        body = self._serialize_packet(packet)
        return EncodedFragment(
            block_id=block_id,
            total_blocks=total_blocks,
            copy_id=copy_id,
            dna=marker + body,
        )

    def _ecc_encode(self, packet: bytes) -> bytes:
        if self._ecc_mode == "none":
            return packet
        if self._ecc_mode == "reed_solomon":
            if self.config.ecc_symbols <= 0:
                raise ValueError("ecc_symbols must be positive for reed_solomon.")
            return _reed_solomon_encode(packet, self.config.ecc_symbols)
        raise ValueError(f"Unsupported ECC mode: {self.config.ecc_mode}")

    def _ecc_decode(self, raw: bytes, erase_positions: list[int] | None = None) -> bytes:
        if self._ecc_mode == "none":
            if erase_positions:
                raise ValueError("Cannot recover chunk erasures without an ECC mode.")
            return raw
        if self._ecc_mode == "reed_solomon":
            if self.config.ecc_symbols <= 0:
                raise ValueError("ecc_symbols must be positive for reed_solomon.")
            return _reed_solomon_decode(raw, self.config.ecc_symbols, erase_positions)
        raise ValueError(f"Unsupported ECC mode: {self.config.ecc_mode}")

    def _serialize_packet(self, packet: bytes) -> str:
        if self._resync_mode == "none":
            return bytes_to_dna(packet)
        if self._resync_mode != "chunked":
            raise ValueError(f"Unsupported resync mode: {self.config.resync_mode}")

        chunk_count = max(1, math.ceil(len(packet) / self.config.resync_chunk_bytes))
        if chunk_count > 65_535:
            raise ValueError("Too many resync chunks for the current chunk header.")

        marker = self._chunk_marker()
        body_parts: list[str] = []
        for chunk_id in range(chunk_count):
            start = chunk_id * self.config.resync_chunk_bytes
            chunk = packet[start : start + self.config.resync_chunk_bytes]
            chunk_header = CHUNK_HEADER.pack(chunk_id, chunk_count, len(chunk), len(packet))
            body_parts.append(marker)
            body_parts.append(bytes_to_dna(chunk_header + chunk))
        return "".join(body_parts)

    def _deserialize_body(self, body: str) -> tuple[bytes, list[int]]:
        if self._resync_mode == "none":
            return dna_to_bytes(body), []
        if self._resync_mode != "chunked":
            raise ValueError(f"Unsupported resync mode: {self.config.resync_mode}")
        return self._deserialize_chunked_body(body)

    def _deserialize_chunked_body(self, body: str) -> tuple[bytes, list[int]]:
        marker = self._chunk_marker()
        positions = self._find_chunk_markers(body, marker)
        if not positions:
            raise ValueError("No chunk markers found.")

        chunks: dict[int, bytes] = {}
        chunk_count: int | None = None
        packet_len: int | None = None
        for index, marker_pos in enumerate(positions):
            segment_start = marker_pos + len(marker)
            segment_end = positions[index + 1] if index + 1 < len(positions) else len(body)
            segment = body[segment_start:segment_end]
            if len(segment) % 4 != 0:
                continue
            try:
                segment_bytes = dna_to_bytes(segment)
            except (KeyError, ValueError):
                continue
            if len(segment_bytes) < CHUNK_HEADER.size:
                continue
            try:
                chunk_id, total_chunks, chunk_len, total_packet_len = CHUNK_HEADER.unpack(
                    segment_bytes[: CHUNK_HEADER.size]
                )
            except struct.error:
                continue
            if total_chunks == 0 or chunk_id >= total_chunks:
                continue
            if chunk_len > self.config.resync_chunk_bytes:
                continue
            expected_end = CHUNK_HEADER.size + chunk_len
            if len(segment_bytes) < expected_end:
                continue
            if chunk_count is None:
                chunk_count = total_chunks
                packet_len = total_packet_len
            if chunk_count != total_chunks or packet_len != total_packet_len:
                continue
            chunks.setdefault(chunk_id, segment_bytes[CHUNK_HEADER.size:expected_end])

        if chunk_count is None or packet_len is None:
            raise ValueError("No valid chunk headers found.")

        raw = bytearray(packet_len)
        erase_positions: list[int] = []
        for chunk_id in range(chunk_count):
            start = chunk_id * self.config.resync_chunk_bytes
            if start >= packet_len:
                continue
            expected_len = min(self.config.resync_chunk_bytes, packet_len - start)
            chunk = chunks.get(chunk_id)
            if chunk is None or len(chunk) != expected_len:
                erase_positions.extend(range(start, start + expected_len))
                continue
            raw[start : start + expected_len] = chunk
        return bytes(raw), erase_positions

    def _chunk_marker(self) -> str:
        material = b""
        counter = 0
        while len(material) * 4 < self.config.resync_marker_bp:
            material += hmac.new(
                self._marker_key,
                b"chunk-sync-v1" + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            counter += 1
        return bytes_to_dna(material)[: self.config.resync_marker_bp]

    def _find_chunk_markers(self, body: str, marker: str) -> list[int]:
        window = len(marker)
        max_mismatches = self.config.resync_mismatches
        positions: list[int] = []
        index = 0
        while index <= len(body) - window:
            candidate = body[index : index + window]
            if _hamming(candidate, marker) <= max_mismatches:
                positions.append(index)
                index += window
                continue
            index += 1
        return positions

    def _validate_block_count(self, total_blocks: int) -> None:
        if total_blocks > 65_535:
            raise ValueError("Too many payload blocks for the current 16-bit block header.")
        if self._erasure_mode in {"xor_parity", "fountain"}:
            if self.config.erasure_group_size <= 0:
                raise ValueError(f"erasure_group_size must be positive for {self._erasure_mode}.")
            parity_groups = math.ceil(total_blocks / self.config.erasure_group_size)
            if self._erasure_mode == "xor_parity":
                repair_blocks = parity_groups
            else:
                if self.config.erasure_group_size > 255:
                    raise ValueError("erasure_group_size cannot exceed 255 for fountain.")
                if self.config.erasure_repair_blocks <= 0:
                    raise ValueError("erasure_repair_blocks must be positive for fountain.")
                if self.config.erasure_repair_blocks > 255:
                    raise ValueError("erasure_repair_blocks cannot exceed 255 for fountain.")
                repair_blocks = parity_groups * self.config.erasure_repair_blocks
            if total_blocks + repair_blocks > 65_535:
                raise ValueError("Too many data+repair blocks for the current 16-bit block header.")

    def _erasure_repair_blocks(self, plain_blocks: list[bytes], total_blocks: int) -> list[tuple[int, bytes]]:
        if self._erasure_mode == "xor_parity":
            return [
                (total_blocks + group_id, parity_plain)
                for group_id, parity_plain in enumerate(self._xor_parity_blocks(plain_blocks))
            ]
        if self._erasure_mode == "fountain":
            return self._fountain_repair_blocks(plain_blocks, total_blocks)
        return []

    def _xor_parity_blocks(self, plain_blocks: list[bytes]) -> list[bytes]:
        if self.config.erasure_group_size <= 0:
            raise ValueError("erasure_group_size must be positive for xor_parity.")
        parity_blocks: list[bytes] = []
        for group_start in range(0, len(plain_blocks), self.config.erasure_group_size):
            parity = bytearray(self.config.block_size_bytes)
            for block in plain_blocks[group_start : group_start + self.config.erasure_group_size]:
                padded = block.ljust(self.config.block_size_bytes, b"\0")
                for index, value in enumerate(padded):
                    parity[index] ^= value
            parity_blocks.append(bytes(parity))
        return parity_blocks

    def _recover_xor_parity_blocks(
        self,
        blocks: dict[int, DecodedBlock],
        repair_blocks: dict[tuple[int, int], DecodedBlock],
        total_blocks: int,
        payload_length: int,
    ) -> None:
        if self._erasure_mode != "xor_parity" or self.config.erasure_group_size <= 0:
            return

        for (group_id, repair_id), parity in repair_blocks.items():
            if repair_id != 0:
                continue
            group_start = group_id * self.config.erasure_group_size
            if group_start >= total_blocks:
                continue
            group_end = min(total_blocks, group_start + self.config.erasure_group_size)
            missing = [block_id for block_id in range(group_start, group_end) if block_id not in blocks]
            if len(missing) != 1:
                continue

            recovered = bytearray(parity.plain.ljust(self.config.block_size_bytes, b"\0")[: self.config.block_size_bytes])
            for block_id in range(group_start, group_end):
                if block_id == missing[0]:
                    continue
                padded = blocks[block_id].plain.ljust(self.config.block_size_bytes, b"\0")
                for index, value in enumerate(padded):
                    recovered[index] ^= value

            blocks[missing[0]] = DecodedBlock(
                block_id=missing[0],
                total_blocks=total_blocks,
                payload_length=payload_length,
                plain=bytes(recovered),
            )

    def _fountain_repair_blocks(self, plain_blocks: list[bytes], total_blocks: int) -> list[tuple[int, bytes]]:
        repair_blocks: list[tuple[int, bytes]] = []
        repair_count = self.config.erasure_repair_blocks
        for group_id, group_start in enumerate(range(0, len(plain_blocks), self.config.erasure_group_size)):
            group = plain_blocks[group_start : group_start + self.config.erasure_group_size]
            padded_group = [block.ljust(self.config.block_size_bytes, b"\0") for block in group]
            for repair_id in range(repair_count):
                coefficients = _fountain_coefficients(len(group), repair_id)
                repair = bytearray(self.config.block_size_bytes)
                for coefficient, block in zip(coefficients, padded_group):
                    if coefficient == 0:
                        continue
                    for index, value in enumerate(block):
                        repair[index] ^= _gf256_mul(coefficient, value)
                block_id = total_blocks + group_id * repair_count + repair_id
                repair_blocks.append((block_id, bytes(repair)))
        return repair_blocks

    def _recover_fountain_blocks(
        self,
        blocks: dict[int, DecodedBlock],
        repair_blocks: dict[tuple[int, int], DecodedBlock],
        total_blocks: int,
        payload_length: int,
    ) -> None:
        if self._erasure_mode != "fountain" or self.config.erasure_group_size <= 0:
            return

        repair_count = self.config.erasure_repair_blocks
        for group_id in range(math.ceil(total_blocks / self.config.erasure_group_size)):
            group_start = group_id * self.config.erasure_group_size
            group_end = min(total_blocks, group_start + self.config.erasure_group_size)
            group_len = group_end - group_start
            if group_len <= 0:
                continue
            if all(block_id in blocks for block_id in range(group_start, group_end)):
                continue

            rows: list[list[int]] = []
            values: list[bytes] = []
            for block_id in range(group_start, group_end):
                if block_id not in blocks:
                    continue
                row = [0] * group_len
                row[block_id - group_start] = 1
                rows.append(row)
                values.append(blocks[block_id].plain.ljust(self.config.block_size_bytes, b"\0")[: self.config.block_size_bytes])

            for repair_id in range(repair_count):
                repair = repair_blocks.get((group_id, repair_id))
                if repair is None:
                    continue
                rows.append(_fountain_coefficients(group_len, repair_id))
                values.append(repair.plain.ljust(self.config.block_size_bytes, b"\0")[: self.config.block_size_bytes])

            solved = _solve_gf256_linear_system(rows, values, group_len, self.config.block_size_bytes)
            if solved is None:
                continue

            for offset, plain in enumerate(solved):
                block_id = group_start + offset
                blocks.setdefault(
                    block_id,
                    DecodedBlock(
                        block_id=block_id,
                        total_blocks=total_blocks,
                        payload_length=payload_length,
                        plain=plain,
                    ),
                )

    def _repair_key(self, block_id: int, total_blocks: int) -> tuple[int, int] | None:
        if block_id < total_blocks:
            return None
        offset = block_id - total_blocks
        if self._erasure_mode == "xor_parity":
            return (offset, 0)
        if self._erasure_mode == "fountain":
            if self.config.erasure_repair_blocks <= 0:
                return None
            return (offset // self.config.erasure_repair_blocks, offset % self.config.erasure_repair_blocks)
        return None


class ExpectedMarkerPayloadCodec(PayloadCodec):
    def __init__(self, config: PayloadCodecConfig, marker: str) -> None:
        super().__init__(config)
        self._expected_marker = marker

    def _find_marker(self, dna: str) -> int:
        marker = self._expected_marker
        window = len(marker)
        max_mismatches = self.config.sync_mismatches
        for index in range(0, max(0, len(dna) - window) + 1):
            candidate = dna[index : index + window]
            if len(candidate) != window:
                continue
            if _hamming(candidate, marker) <= max_mismatches:
                return index
        return -1


def normalize_encryption_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "hmac_stream": "hmac_stream",
        "simulation": "hmac_stream",
        "sim": "hmac_stream",
        "chacha20_poly1305": "chacha20_poly1305",
        "chacha20poly1305": "chacha20_poly1305",
        "aead_chacha20_poly1305": "chacha20_poly1305",
        "aead": "chacha20_poly1305",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported encryption mode: {mode!r}")
    return aliases[normalized]


def normalize_ecc_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "no_ecc": "none",
        "reed_solomon": "reed_solomon",
        "reedsolomon": "reed_solomon",
        "rs": "reed_solomon",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported ECC mode: {mode!r}")
    return aliases[normalized]


def normalize_erasure_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "no_erasure": "none",
        "xor": "xor_parity",
        "xor_parity": "xor_parity",
        "parity": "xor_parity",
        "fountain": "fountain",
        "fountain_rs": "fountain",
        "rs_erasure": "fountain",
        "reed_solomon_erasure": "fountain",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported erasure mode: {mode!r}")
    return aliases[normalized]


def normalize_resync_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "no_resync": "none",
        "chunked": "chunked",
        "chunked_sync": "chunked",
        "chunk_sync": "chunked",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported resync mode: {mode!r}")
    return aliases[normalized]


def _chacha20_poly1305_encrypt(key: bytes, nonce: bytes, plain: bytes, associated_data: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "cryptography is required for encryption_mode='chacha20_poly1305'. "
            "Install it with: python -m pip install cryptography"
        ) from exc
    return ChaCha20Poly1305(key).encrypt(nonce, plain, associated_data)


def _chacha20_poly1305_decrypt(key: bytes, nonce: bytes, encrypted: bytes, associated_data: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "cryptography is required for encryption_mode='chacha20_poly1305'. "
            "Install it with: python -m pip install cryptography"
        ) from exc
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, encrypted, associated_data)
    except Exception as exc:
        raise ValueError("Authentication failed.") from exc


def _reed_solomon_encode(packet: bytes, ecc_symbols: int) -> bytes:
    try:
        from reedsolo import RSCodec
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "reedsolo is required for ecc_mode='reed_solomon'. "
            "Install it with: python -m pip install reedsolo"
        ) from exc
    return bytes(RSCodec(ecc_symbols).encode(packet))


def _reed_solomon_decode(raw: bytes, ecc_symbols: int, erase_positions: list[int] | None = None) -> bytes:
    try:
        from reedsolo import ReedSolomonError, RSCodec
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "reedsolo is required for ecc_mode='reed_solomon'. "
            "Install it with: python -m pip install reedsolo"
        ) from exc
    try:
        decoded = RSCodec(ecc_symbols).decode(raw, erase_pos=erase_positions or [])
    except ReedSolomonError as exc:
        raise ValueError("Reed-Solomon correction failed.") from exc
    if isinstance(decoded, tuple):
        return bytes(decoded[0])
    return bytes(decoded)


def _fountain_coefficients(variable_count: int, repair_id: int) -> list[int]:
    x = repair_id + 1
    value = 1
    coefficients: list[int] = []
    for _ in range(variable_count):
        coefficients.append(value)
        value = _gf256_mul(value, x)
    return coefficients


def _solve_gf256_linear_system(
    rows: list[list[int]],
    values: list[bytes],
    variable_count: int,
    value_size: int,
) -> list[bytes] | None:
    if len(rows) != len(values):
        raise ValueError("GF(256) solver requires one value vector per row.")
    if variable_count <= 0:
        return []

    matrix = [row[:] for row in rows]
    rhs = [bytearray(value.ljust(value_size, b"\0")[:value_size]) for value in values]
    pivot_by_col: dict[int, int] = {}
    pivot_row = 0

    for col in range(variable_count):
        pivot = None
        for row_index in range(pivot_row, len(matrix)):
            if matrix[row_index][col] != 0:
                pivot = row_index
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            rhs[pivot_row], rhs[pivot] = rhs[pivot], rhs[pivot_row]

        inverse = _gf256_inv(matrix[pivot_row][col])
        for c in range(col, variable_count):
            matrix[pivot_row][c] = _gf256_mul(matrix[pivot_row][c], inverse)
        for index, value in enumerate(rhs[pivot_row]):
            rhs[pivot_row][index] = _gf256_mul(value, inverse)

        for row_index in range(len(matrix)):
            if row_index == pivot_row:
                continue
            factor = matrix[row_index][col]
            if factor == 0:
                continue
            for c in range(col, variable_count):
                matrix[row_index][c] ^= _gf256_mul(factor, matrix[pivot_row][c])
            for index, value in enumerate(rhs[pivot_row]):
                rhs[row_index][index] ^= _gf256_mul(factor, value)

        pivot_by_col[col] = pivot_row
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    if len(pivot_by_col) < variable_count:
        return None
    return [bytes(rhs[pivot_by_col[col]]) for col in range(variable_count)]


def _gf256_tables() -> tuple[list[int], list[int]]:
    exp = [0] * 512
    log = [0] * 256
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    for index in range(255, 512):
        exp[index] = exp[index - 255]
    return exp, log


GF256_EXP, GF256_LOG = _gf256_tables()


def _gf256_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return GF256_EXP[GF256_LOG[left] + GF256_LOG[right]]


def _gf256_inv(value: int) -> int:
    if value == 0:
        raise ZeroDivisionError("Cannot invert zero in GF(256).")
    return GF256_EXP[255 - GF256_LOG[value]]
