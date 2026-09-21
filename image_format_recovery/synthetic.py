from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from random import Random
from typing import Sequence

from PIL import Image, ImageDraw

from .formats import FORMAT_NAMES, normalize_format


@dataclass(frozen=True)
class SyntheticSample:
    stream: bytes
    payload: bytes
    format_name: str
    start_offset: int
    width: int
    height: int


def available_formats(requested: Sequence[str] | None = None) -> tuple[str, ...]:
    formats = tuple(normalize_format(item) for item in (requested or FORMAT_NAMES))
    result: list[str] = []
    for format_name in formats:
        if format_name == "webp" and "WEBP" not in Image.registered_extensions().values():
            continue
        result.append(format_name)
    return tuple(dict.fromkeys(result))


def make_pattern_image(rng: Random, width: int = 96, height: int = 96) -> Image.Image:
    """Create a deterministic synthetic image without external data."""

    base = Image.new("RGB", (width, height), (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(base)

    for y in range(height):
        color = (
            (y * 3 + rng.randrange(80)) % 256,
            (y * 5 + rng.randrange(80)) % 256,
            (y * 7 + rng.randrange(80)) % 256,
        )
        draw.line((0, y, width, y), fill=color)

    for _ in range(12):
        x0 = rng.randrange(width)
        y0 = rng.randrange(height)
        x1 = min(width - 1, x0 + rng.randrange(8, max(9, width // 2)))
        y1 = min(height - 1, y0 + rng.randrange(8, max(9, height // 2)))
        color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        if rng.random() < 0.5:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=rng.randrange(1, 4))
        else:
            draw.ellipse((x0, y0, x1, y1), outline=color, width=rng.randrange(1, 4))

    for _ in range(30):
        x = rng.randrange(width)
        y = rng.randrange(height)
        draw.point((x, y), fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))

    return base


def encode_image(image: Image.Image, format_name: str, quality: int = 90) -> bytes:
    normalized = normalize_format(format_name)
    pil_format = {
        "jpeg": "JPEG",
        "png": "PNG",
        "bmp": "BMP",
        "webp": "WEBP",
        "gif": "GIF",
        "tiff": "TIFF",
    }[normalized]

    output = BytesIO()
    save_kwargs = {}
    if normalized in {"jpeg", "webp"}:
        save_kwargs["quality"] = quality
    if normalized == "jpeg":
        image = image.convert("RGB")
    image.save(output, format=pil_format, **save_kwargs)
    return output.getvalue()


def random_bytes(rng: Random, length: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(length))


def make_sample(
    rng: Random,
    format_name: str,
    max_prefix: int = 256,
    max_suffix: int = 128,
    width: int = 96,
    height: int = 96,
) -> SyntheticSample:
    image = make_pattern_image(rng, width=width, height=height)
    payload = encode_image(image, format_name=format_name, quality=rng.randrange(70, 96))
    prefix = random_bytes(rng, rng.randrange(max_prefix + 1))
    suffix = random_bytes(rng, rng.randrange(max_suffix + 1))
    stream = prefix + payload + suffix
    return SyntheticSample(
        stream=stream,
        payload=payload,
        format_name=normalize_format(format_name),
        start_offset=len(prefix),
        width=width,
        height=height,
    )
