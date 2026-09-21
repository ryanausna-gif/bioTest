"""Run with python -m genome_mining.rrc_cli --help."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import secrets
from pathlib import Path

from .lm_stego.model_adapter import KmerProbabilityModel
from .lm_stego.rrc import RRCConfig, encode_packet, decode_packet, sample_dna
from .lm_stego.rrc_models import probability_source


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def context_records(path, size):
    """Stream nonoverlapping windows, including gzip, without loading chromosomes."""
    opener = gzip.open if str(path).endswith(".gz") else open
    chrom, buffer, start = None, "", 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                chrom, buffer, start = line[1:].split()[0], "", 0
            elif line:
                if chrom is None:
                    raise ValueError("FASTA sequence before header")
                buffer += line.upper()
                while len(buffer) >= size:
                    yield chrom, start, buffer[:size]
                    start += size
                    buffer = buffer[size:]


def prepare(args):
    from .rrc_data import prepare_contexts
    prepare_contexts(args)


def audit_rows(rows, length=None):
    from .rrc_data import audit_contexts
    return audit_contexts(rows, length)


def build(args):
    from .rrc_experiment import build_experiment
    build_experiment(args)


def evaluate(args):
    from .rrc_evaluation import evaluate_experiment
    evaluate_experiment(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    key = commands.add_parser("keygen")
    key.add_argument("--out", required=True)
    prep = commands.add_parser("prepare-contexts")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--per-donor", type=int, default=10)
    prep.add_argument("--length", type=int, default=512)
    prep.add_argument("--context-bases", type=int, default=1024)
    prep.add_argument("--sampling", choices=["reservoir", "first"], default="reservoir")
    prep.add_argument("--seed", type=int, default=2026)
    fit = commands.add_parser("fit-context-model")
    fit.add_argument("--contexts", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--order", type=int, default=4)
    for name in ("encode", "decode", "sample", "build", "verify"):
        p = commands.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--model", help="Serialized k-mer model")
        source.add_argument("--hf-model", help="Local HF causal DNA checkpoint directory")
        p.add_argument("--device", default="cpu")
        p.add_argument("--trust-remote-code", action="store_true")
        p.add_argument("--key-file", required=True)
        if name != "verify":
            p.add_argument("--out", required=True)
        p.add_argument("--context-bases", type=int, default=1024)
        p.add_argument("--precision-bits", type=int, default=12)
        p.add_argument("--max-bases", type=int, default=4096)
        if name == "build":
            p.add_argument("--contexts", required=True)
            p.add_argument("--bits", type=int, default=128)
            p.add_argument("--length", type=int, default=512)
            p.add_argument("--resume", action="store_true")
        elif name == "verify":
            p.add_argument("--dataset", required=True)
            p.add_argument("--contexts", required=True)
        elif name == "decode":
            p.add_argument("--dna", required=True)
            p.add_argument("--metadata", required=True)
        else:
            p.add_argument("--context", default="ACGT")
            p.add_argument("--length", type=int, default=512)
            if name == "encode":
                p.add_argument("--bits", default="01010101", help="Literal binary message")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--csv", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--fpr", type=float, default=0.01)
    ev.add_argument("--detector", choices=["logreg", "cnn", "hyena-probe"], default="logreg")
    ev.add_argument("--hf-model", help="Local causal HF checkpoint for frozen Hyena embeddings")
    ev.add_argument("--trust-remote-code", action="store_true")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--epochs", type=int, default=10)
    ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--bootstrap", type=int, default=500)
    ev.add_argument("--allow-failures", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--detector-dir", required=True)
    score.add_argument("--csv", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--hf-model")
    score.add_argument("--trust-remote-code", action="store_true")
    transfer = commands.add_parser("evaluate-transfer")
    transfer.add_argument("--detector-dir", required=True)
    transfer.add_argument("--csv", required=True)
    transfer.add_argument("--out", required=True)
    transfer.add_argument("--device", default="cpu")
    transfer.add_argument("--batch-size", type=int, default=32)
    transfer.add_argument("--hf-model")
    transfer.add_argument("--trust-remote-code", action="store_true")
    transfer.add_argument("--bootstrap", type=int, default=500)
    transfer.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "keygen":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(secrets.token_bytes(32))
    elif args.command == "prepare-contexts":
        prepare(args)
    elif args.command == "fit-context-model":
        rows = read_csv(args.contexts)
        audit_rows(rows)
        train = [r["context"] + r["natural"] for r in rows if r["split"] == "train"]
        if not train or Path(args.out).exists():
            raise ValueError("Need train rows and a new output path")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        KmerProbabilityModel.fit(train, order=args.order).save(args.out)
    elif args.command == "build":
        build(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "verify":
        from .rrc_experiment import verify_experiment
        verify_experiment(args)
    elif args.command == "score":
        from .rrc_evaluation import score_saved
        score_saved(args)
    elif args.command == "evaluate-transfer":
        from .rrc_evaluation import evaluate_transfer
        evaluate_transfer(args)
    else:
        model, key_bytes = probability_source(args), Path(args.key_file).read_bytes()
        out = new_output(args.out)
        config = RRCConfig(args.precision_bits, args.context_bases, args.max_bases)
        if args.command == "decode":
            dna = Path(args.dna).read_text().strip()
            meta = json.loads(Path(args.metadata).read_text())
            bits = decode_packet(dna, meta, model, key=key_bytes)
            (out / "message.bits").write_text(bits + "\n")
        elif args.command == "sample":
            nonce = secrets.token_bytes(16)
            dna = sample_dna(args.length, model, key=key_bytes, nonce=nonce,
                             context=args.context, config=config)
            (out / "ordinary.dna.txt").write_text(dna + "\n")
            write_json(out / "sampling.json", {"nonce": nonce.hex(), "context": args.context,
                       "config": vars(config), "model": model.metadata(), "length": args.length})
        else:
            dna, meta = encode_packet(args.bits, model, key=key_bytes, nonce=secrets.token_bytes(16),
                                      context=args.context, config=config, output_bases=args.length)
            if decode_packet(dna, meta, model, key=key_bytes) != args.bits:
                raise ValueError("Self-check failed")
            (out / "carrier.dna.txt").write_text(dna + "\n")
            write_json(out / "receiver.json", meta)
            print(json.dumps({"exact": True, **meta["metrics"]}))


if __name__ == "__main__":
    main()
