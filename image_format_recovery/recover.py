from __future__ import annotations

import argparse
from pathlib import Path

from .formats import recover_file, recover_image


def _predict_with_checkpoint(checkpoint_path: Path, data: bytes) -> tuple[str, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
        raise SystemExit(
            "PyTorch is required when --checkpoint is used. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    from .model import ByteCNNTransformer

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    formats = tuple(checkpoint["formats"])
    seq_len = int(checkpoint["seq_len"])
    max_prefix = int(checkpoint["max_prefix"])

    model = ByteCNNTransformer(num_formats=len(formats), max_prefix=max_prefix, seq_len=seq_len)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokens = list(data[:seq_len])
    if len(tokens) < seq_len:
        tokens.extend([256] * (seq_len - len(tokens)))

    with torch.no_grad():
        byte_tokens = torch.tensor([tokens], dtype=torch.long)
        output = model(byte_tokens)
        format_id = int(output["format_logits"].argmax(dim=-1).item())
        start = int(output["start_logits"].argmax(dim=-1).item())

    return formats[format_id], start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover an image payload from an unknown byte stream.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format-hint", default=None)
    parser.add_argument("--start-hint", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    format_hint = args.format_hint
    start_hint = args.start_hint

    if args.checkpoint is not None:
        data = args.input.read_bytes()
        format_hint, start_hint = _predict_with_checkpoint(args.checkpoint, data)
        image, candidate = recover_image(data, format_hint=format_hint, start_hint=start_hint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
    else:
        candidate = recover_file(args.input, args.output, format_hint=format_hint, start_hint=start_hint)

    if candidate is None:
        print("Recovered image by direct decoding.")
    else:
        print(
            "Recovered image "
            f"format={candidate.format_name} "
            f"start={candidate.start} "
            f"end={candidate.end} "
            f"reason={candidate.reason}"
        )


if __name__ == "__main__":
    main()
