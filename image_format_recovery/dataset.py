from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Sequence

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ImportError(
        "PyTorch is required for image_format_recovery.dataset. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from .synthetic import SyntheticSample, available_formats, encode_image, make_sample, random_bytes


IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _tokens_from_stream(stream: bytes, seq_len: int) -> list[int]:
    tokens = list(stream[:seq_len])
    if len(tokens) < seq_len:
        tokens.extend([256] * (seq_len - len(tokens)))
    return tokens


def _sample_to_tensors(
    sample: SyntheticSample,
    format_to_id: dict[str, int],
    seq_len: int,
    max_prefix: int,
) -> dict[str, torch.Tensor]:
    return {
        "bytes": torch.tensor(_tokens_from_stream(sample.stream, seq_len), dtype=torch.long),
        "format": torch.tensor(format_to_id[sample.format_name], dtype=torch.long),
        "start": torch.tensor(min(sample.start_offset, max_prefix), dtype=torch.long),
    }


class SyntheticImageByteDataset(Dataset):
    """On-the-fly byte-stream dataset for format classification and offset prediction."""

    def __init__(
        self,
        size: int,
        formats: Sequence[str] | None = None,
        seq_len: int = 4096,
        max_prefix: int = 256,
        max_suffix: int = 128,
        seed: int = 13,
    ) -> None:
        self.size = size
        self.formats = available_formats(formats)
        if not self.formats:
            raise ValueError("No supported image formats are available.")
        self.seq_len = seq_len
        self.max_prefix = max_prefix
        self.max_suffix = max_suffix
        self.seed = seed
        self.format_to_id = {name: index for index, name in enumerate(self.formats)}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = Random(self.seed + index * 7919)
        format_name = self.formats[rng.randrange(len(self.formats))]
        sample = make_sample(
            rng,
            format_name=format_name,
            max_prefix=self.max_prefix,
            max_suffix=self.max_suffix,
        )

        return _sample_to_tensors(sample, self.format_to_id, self.seq_len, self.max_prefix)


class ImageFolderByteDataset(Dataset):
    """Byte-stream dataset backed by real image files.

    Source images are used as image content. Each item re-encodes one source
    image into a randomly selected target format, then wraps it with random
    prefix/suffix bytes to simulate an unknown container stream.
    """

    def __init__(
        self,
        data_dir: str | Path,
        size: int,
        formats: Sequence[str] | None = None,
        seq_len: int = 4096,
        max_prefix: int = 256,
        max_suffix: int = 128,
        image_size: int = 128,
        seed: int = 13,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.paths = sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No supported images found under {self.data_dir}")

        self.size = size
        self.formats = available_formats(formats)
        if not self.formats:
            raise ValueError("No supported image formats are available.")
        self.seq_len = seq_len
        self.max_prefix = max_prefix
        self.max_suffix = max_suffix
        self.image_size = image_size
        self.seed = seed
        self.format_to_id = {name: index for index, name in enumerate(self.formats)}

    def __len__(self) -> int:
        return self.size

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.image_size > 0:
                image.thumbnail((self.image_size, self.image_size))
                canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
                left = (self.image_size - image.width) // 2
                top = (self.image_size - image.height) // 2
                canvas.paste(image, (left, top))
                return canvas
            return image.copy()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = Random(self.seed + index * 7919)
        path = self.paths[rng.randrange(len(self.paths))]
        format_name = self.formats[rng.randrange(len(self.formats))]
        image = self._load_image(path)
        payload = encode_image(image, format_name=format_name, quality=rng.randrange(70, 96))
        prefix = random_bytes(rng, rng.randrange(self.max_prefix + 1))
        suffix = random_bytes(rng, rng.randrange(self.max_suffix + 1))
        sample = SyntheticSample(
            stream=prefix + payload + suffix,
            payload=payload,
            format_name=format_name,
            start_offset=len(prefix),
            width=image.width,
            height=image.height,
        )
        return _sample_to_tensors(sample, self.format_to_id, self.seq_len, self.max_prefix)
