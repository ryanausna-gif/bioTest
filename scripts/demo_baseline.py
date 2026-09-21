from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_format_recovery.formats import recover_image
from image_format_recovery.synthetic import available_formats, make_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noisy image byte streams and recover them.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--formats", nargs="*", default=["png", "jpeg", "bmp", "webp"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(args.seed)
    formats = available_formats(args.formats)

    for format_name in formats:
        sample = make_sample(rng, format_name=format_name, max_prefix=128, max_suffix=64)
        stream_path = args.out_dir / f"sample_{format_name}.bin"
        output_path = args.out_dir / f"recovered_{format_name}.png"
        stream_path.write_bytes(sample.stream)

        image, candidate = recover_image(sample.stream)
        image.save(output_path)
        print(
            f"{format_name}: expected_start={sample.start_offset} "
            f"detected_start={candidate.start if candidate else 0} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
