from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path


PAIR_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
BASE_TO_PAIR = {base: value for value, base in PAIR_TO_BASE.items()}
BYTE_TO_DNA = tuple(
    "".join(PAIR_TO_BASE[(value >> shift) & 0b11] for shift in (6, 4, 2, 0))
    for value in range(256)
)


@dataclass(frozen=True)
class RawImageDNASpec:
    width: int = 128
    height: int = 128
    channels: int = 1
    bits_per_channel: int = 8
    bases_per_byte: int = 4
    storage_order: str = "row_major"
    color_mode: str = "L"
    codec_name: str = "raw_gray_2bit_v1"
    resize_short_side: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.channels != 1 or self.bits_per_channel != 8:
            raise ValueError("The MVP supports only one-channel 8-bit images.")
        if self.bases_per_byte != 4:
            raise ValueError("The MVP uses four DNA bases per byte.")
        if self.resize_short_side is not None and self.resize_short_side <= 0:
            raise ValueError("resize_short_side must be positive when provided.")

    @property
    def payload_bytes(self) -> int:
        return self.width * self.height * self.channels

    @property
    def payload_bases(self) -> int:
        return self.payload_bytes * self.bases_per_byte

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "payload_bytes": self.payload_bytes, "payload_bases": self.payload_bases}


def preprocess_image(path: str | Path, spec: RawImageDNASpec):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError("Pillow is required for image-to-DNA preprocessing.") from exc

    with Image.open(path) as source:
        grayscale = ImageOps.exif_transpose(source).convert(spec.color_mode)
        if spec.resize_short_side is not None:
            scale = spec.resize_short_side / min(grayscale.size)
            resized_size = tuple(max(1, round(value * scale)) for value in grayscale.size)
            resized = grayscale.resize(resized_size, resample=Image.Resampling.LANCZOS)
            if resized.width < spec.width or resized.height < spec.height:
                raise ValueError(
                    "resize_short_side produces an image smaller than the requested center crop."
                )
            left = (resized.width - spec.width) // 2
            top = (resized.height - spec.height) // 2
            return resized.crop((left, top, left + spec.width, top + spec.height))
        return ImageOps.fit(
            grayscale,
            (spec.width, spec.height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def bytes_to_dna(payload: bytes) -> str:
    return "".join(BYTE_TO_DNA[value] for value in payload)


def dna_to_bytes(sequence: str, *, expected_bytes: int | None = None) -> bytes:
    dna = str(sequence).upper()
    if len(dna) % 4:
        raise ValueError("DNA payload length must be divisible by four.")
    if any(base not in BASE_TO_PAIR for base in dna):
        raise ValueError("Raw image DNA payload may contain only A/C/G/T.")
    output = bytearray()
    for index in range(0, len(dna), 4):
        value = 0
        for base in dna[index : index + 4]:
            value = (value << 2) | BASE_TO_PAIR[base]
        output.append(value)
    if expected_bytes is not None and len(output) != expected_bytes:
        raise ValueError(f"Decoded {len(output)} bytes; expected {expected_bytes}.")
    return bytes(output)


def encode_image_file(path: str | Path, spec: RawImageDNASpec) -> tuple[str, bytes]:
    image = preprocess_image(path, spec)
    pixels = image.tobytes()
    if len(pixels) != spec.payload_bytes:
        raise ValueError(f"Preprocessed image has {len(pixels)} bytes; expected {spec.payload_bytes}.")
    return bytes_to_dna(pixels), pixels


def decode_image_dna(sequence: str, spec: RawImageDNASpec):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for DNA-to-image decoding.") from exc

    if len(sequence) != spec.payload_bases:
        raise ValueError(f"Image DNA span has {len(sequence)} bases; expected {spec.payload_bases}.")
    pixels = dna_to_bytes(sequence, expected_bytes=spec.payload_bytes)
    return Image.frombytes(spec.color_mode, (spec.width, spec.height), pixels)


def compare_pixel_bytes(reference: bytes, recovered: bytes) -> dict[str, float | bool]:
    if len(reference) != len(recovered):
        raise ValueError("Pixel arrays must have equal lengths.")
    if not reference:
        return {
            "exact": True,
            "pixel_exact_fraction": 1.0,
            "pixel_mae": 0.0,
            "pixel_mse": 0.0,
            "psnr": float("inf"),
            "ssim_global": 1.0,
        }
    differences = [int(left) - int(right) for left, right in zip(reference, recovered)]
    exact_count = sum(diff == 0 for diff in differences)
    mae = sum(abs(diff) for diff in differences) / len(differences)
    mse = sum(diff * diff for diff in differences) / len(differences)
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    mean_x = sum(reference) / len(reference)
    mean_y = sum(recovered) / len(recovered)
    var_x = sum((value - mean_x) ** 2 for value in reference) / len(reference)
    var_y = sum((value - mean_y) ** 2 for value in recovered) / len(recovered)
    covariance = sum(
        (left - mean_x) * (right - mean_y) for left, right in zip(reference, recovered)
    ) / len(reference)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return {
        "exact": exact_count == len(differences),
        "pixel_exact_fraction": exact_count / len(differences),
        "pixel_mae": float(mae),
        "pixel_mse": float(mse),
        "psnr": float(psnr),
        "ssim_global": float(ssim),
    }


def dna_to_token_values(sequence: str) -> list[int]:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    return [mapping.get(base, 4) for base in str(sequence).upper()]


def token_values_to_dna(values) -> str:
    alphabet = "ACGTN"
    return "".join(alphabet[int(value)] if 0 <= int(value) < len(alphabet) else "N" for value in values)
