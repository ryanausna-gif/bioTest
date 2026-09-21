from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class PayloadCandidate:
    """A possible image payload found inside a larger byte stream."""

    format_name: str
    start: int
    end: int
    confidence: float
    reason: str

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


FORMAT_NAMES = ("png", "jpeg", "bmp", "webp", "gif", "tiff")


def _find_all(data: bytes, needle: bytes) -> Iterable[int]:
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            break
        yield index
        start = index + 1


def _u32le(data: bytes, offset: int) -> int | None:
    if offset + 4 > len(data):
        return None
    return int.from_bytes(data[offset : offset + 4], "little", signed=False)


def _png_candidates(data: bytes) -> list[PayloadCandidate]:
    signature = b"\x89PNG\r\n\x1a\n"
    end_marker = b"IEND\xaeB`\x82"
    candidates: list[PayloadCandidate] = []
    for start in _find_all(data, signature):
        marker = data.find(end_marker, start + len(signature))
        if marker >= 0:
            end = marker + len(end_marker)
            candidates.append(PayloadCandidate("png", start, end, 0.99, "png signature and IEND marker"))
        else:
            candidates.append(PayloadCandidate("png", start, len(data), 0.65, "png signature only"))
    return candidates


def _jpeg_candidates(data: bytes) -> list[PayloadCandidate]:
    candidates: list[PayloadCandidate] = []
    for start in _find_all(data, b"\xff\xd8\xff"):
        end_marker = data.find(b"\xff\xd9", start + 3)
        if end_marker >= 0:
            end = end_marker + 2
            candidates.append(PayloadCandidate("jpeg", start, end, 0.98, "jpeg SOI and EOI markers"))
        else:
            candidates.append(PayloadCandidate("jpeg", start, len(data), 0.60, "jpeg SOI marker only"))
    return candidates


def _bmp_candidates(data: bytes) -> list[PayloadCandidate]:
    candidates: list[PayloadCandidate] = []
    for start in _find_all(data, b"BM"):
        size = _u32le(data, start + 2)
        if size is None:
            continue
        end = start + size
        if size >= 14 and end <= len(data):
            candidates.append(PayloadCandidate("bmp", start, end, 0.95, "bmp signature and file-size field"))
    return candidates


def _webp_candidates(data: bytes) -> list[PayloadCandidate]:
    candidates: list[PayloadCandidate] = []
    for start in _find_all(data, b"RIFF"):
        if start + 12 > len(data) or data[start + 8 : start + 12] != b"WEBP":
            continue
        riff_size = _u32le(data, start + 4)
        if riff_size is None:
            continue
        end = start + riff_size + 8
        if end <= len(data):
            candidates.append(PayloadCandidate("webp", start, end, 0.96, "RIFF WEBP header and RIFF size"))
    return candidates


def _gif_candidates(data: bytes) -> list[PayloadCandidate]:
    candidates: list[PayloadCandidate] = []
    for signature in (b"GIF87a", b"GIF89a"):
        for start in _find_all(data, signature):
            trailer = data.find(b"\x3b", start + len(signature))
            if trailer >= 0:
                candidates.append(PayloadCandidate("gif", start, trailer + 1, 0.82, "gif signature and trailer"))
            else:
                candidates.append(PayloadCandidate("gif", start, len(data), 0.58, "gif signature only"))
    return candidates


def _tiff_candidates(data: bytes) -> list[PayloadCandidate]:
    candidates: list[PayloadCandidate] = []
    for signature in (b"II*\x00", b"MM\x00*"):
        for start in _find_all(data, signature):
            candidates.append(PayloadCandidate("tiff", start, len(data), 0.60, "tiff signature"))
    return candidates


def detect_payloads(data: bytes, format_hint: str | None = None) -> list[PayloadCandidate]:
    """Find image-like payloads in a byte stream using file signatures."""

    scanners = (
        _png_candidates,
        _jpeg_candidates,
        _bmp_candidates,
        _webp_candidates,
        _gif_candidates,
        _tiff_candidates,
    )
    candidates: list[PayloadCandidate] = []
    for scanner in scanners:
        candidates.extend(scanner(data))

    if format_hint:
        hint = normalize_format(format_hint)
        candidates = [candidate for candidate in candidates if candidate.format_name == hint]

    return sorted(candidates, key=lambda item: (-item.confidence, item.start, item.length))


def normalize_format(name: str) -> str:
    normalized = name.strip().lower().lstrip(".")
    aliases = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "jepg": "jpeg",
        "png": "png",
        "pnj": "png",
        "bmp": "bmp",
        "webp": "webp",
        "gif": "gif",
        "tif": "tiff",
        "tiff": "tiff",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported image format: {name!r}")
    return aliases[normalized]


def decode_payload(payload: bytes) -> Image.Image:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def recover_image(
    data: bytes,
    format_hint: str | None = None,
    start_hint: int | None = None,
) -> tuple[Image.Image, PayloadCandidate | None]:
    """Recover an image from a larger byte stream.

    The function first tries signature-based payload extraction. If no signature
    works, it falls back to direct Pillow decoding from the hinted offset and
    then from the whole stream.
    """

    hints: list[PayloadCandidate] = []
    if start_hint is not None and 0 <= start_hint < len(data):
        end = len(data)
        format_name = normalize_format(format_hint) if format_hint else "unknown"
        hints.append(PayloadCandidate(format_name, start_hint, end, 0.50, "model start hint"))

    candidates = hints + detect_payloads(data, format_hint)
    for candidate in candidates:
        try:
            payload = data[candidate.start : candidate.end]
            return decode_payload(payload), candidate
        except Exception:
            continue

    try:
        return decode_payload(data), None
    except Exception as exc:
        raise ValueError("No decodable image payload found in byte stream.") from exc


def recover_file(
    input_path: str | Path,
    output_path: str | Path,
    format_hint: str | None = None,
    start_hint: int | None = None,
) -> PayloadCandidate | None:
    data = Path(input_path).read_bytes()
    image, candidate = recover_image(data, format_hint=format_hint, start_hint=start_hint)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return candidate
