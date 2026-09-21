from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise SystemExit(
        "PyTorch is required for training. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from .dataset import ImageFolderByteDataset, SyntheticImageByteDataset
from .model import ByteCNNTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level image format recovery model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-prefix", type=int, default=256)
    parser.add_argument("--max-suffix", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-start", type=float, default=0.25)
    parser.add_argument("--formats", nargs="*", default=["png", "jpeg", "bmp", "webp", "gif", "tiff"])
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional directory of real source images.")
    parser.add_argument("--image-size", type=int, default=128, help="Resize real source images to this square size.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_size = args.steps_per_epoch * args.batch_size
    if args.data_dir is None:
        dataset = SyntheticImageByteDataset(
            size=train_size,
            formats=args.formats,
            seq_len=args.seq_len,
            max_prefix=args.max_prefix,
            max_suffix=args.max_suffix,
        )
        data_source = "synthetic"
    else:
        dataset = ImageFolderByteDataset(
            data_dir=args.data_dir,
            size=train_size,
            formats=args.formats,
            seq_len=args.seq_len,
            max_prefix=args.max_prefix,
            max_suffix=args.max_suffix,
            image_size=args.image_size,
        )
        data_source = str(args.data_dir)
        print(f"loaded {len(dataset.paths)} source images from {args.data_dir}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ByteCNNTransformer(
        num_formats=len(dataset.formats),
        max_prefix=args.max_prefix,
        seq_len=args.seq_len,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_format = 0
        total_start = 0
        total = 0

        for batch in loader:
            byte_tokens = batch["bytes"].to(args.device)
            format_target = batch["format"].to(args.device)
            start_target = batch["start"].to(args.device)

            output = model(byte_tokens)
            format_loss = F.cross_entropy(output["format_logits"], format_target)
            start_loss = F.cross_entropy(output["start_logits"], start_target)
            loss = format_loss + args.lambda_start * start_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_size = byte_tokens.shape[0]
                total += batch_size
                total_loss += loss.item() * batch_size
                total_format += (output["format_logits"].argmax(dim=-1) == format_target).sum().item()
                total_start += (output["start_logits"].argmax(dim=-1) == start_target).sum().item()

        print(
            f"epoch={epoch} "
            f"loss={total_loss / total:.4f} "
            f"format_acc={total_format / total:.4f} "
            f"start_acc={total_start / total:.4f}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "formats": dataset.formats,
            "seq_len": args.seq_len,
            "max_prefix": args.max_prefix,
            "data_source": data_source,
        },
        args.checkpoint,
    )
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
