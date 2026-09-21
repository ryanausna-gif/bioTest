"""Resumable paired generation; checkpoints include every attempted message."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections import Counter
from pathlib import Path

from .lm_stego.rrc import (
    PROTOCOL, RRCBudgetError, RRCConfig, canonical, encode_packet, decode_packet, sample_dna,
)
from .lm_stego.rrc_models import probability_source
from .rrc_data import audit_contexts, audit_carriers

EXPERIMENT = "rrc_paired_v2"


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def signed_write(path, data, key):
    signed = {"data": data, "tag": hmac.new(key, b"rrc-checkpoint/" + canonical(data), hashlib.sha256).hexdigest()}
    atomic_json(path, signed)


def signed_read(path, key):
    signed = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = hmac.new(key, b"rrc-checkpoint/" + canonical(signed["data"]), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signed["tag"]):
        raise ValueError("Checkpoint authentication failed")
    return signed["data"]


def pair_material(key, run_id, sample_id, label, length):
    return hashlib.shake_256(hmac.new(key, b"rrc-experiment/" + canonical(
        [run_id, sample_id, label]), hashlib.sha256).digest()).digest(length)


def build_experiment(args):
    from .rrc_cli import read_csv, write_csv
    if not 1 <= args.bits <= 2048 or not 1 <= args.length <= args.max_bases:
        raise ValueError("Use 1..2048 bits and length <= max_bases")
    rows = read_csv(args.contexts)
    audit = audit_contexts(rows, args.length)
    key = Path(args.key_file).read_bytes()
    if len(key) < 32:
        raise ValueError("Key requires at least 32 bytes")
    model = probability_source(args)
    config = RRCConfig(args.precision_bits, args.context_bases, args.max_bases)
    request = {"experiment": EXPERIMENT, "protocol": PROTOCOL, "model": model.metadata(),
               "config": vars(config), "bits": args.bits, "length": args.length,
               "device": str(args.device), "key_fingerprint": hashlib.sha256(key).hexdigest(),
               "contexts_sha256": hashlib.sha256(Path(args.contexts).read_bytes()).hexdigest()}
    out = Path(args.out)
    resume = getattr(args, "resume", False)
    if resume:
        plan = signed_read(out / "receiver_private/plan.json", key)
        if plan["request"] != request:
            raise ValueError("Resume configuration/key/model/input mismatch; use a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "receiver_private").mkdir(mode=0o700)
        plan = {"request": request, "run_id": secrets.token_hex(16)}
        signed_write(out / "receiver_private/plan.json", plan, key)
    private = out / "receiver_private"
    results, metrics, failures = [], [], []
    atomic_json(out / "run.json", {**request, "status": "running", "audit": audit,
                                    "planned": len(rows), "run_id": plan["run_id"]})
    for index, row in enumerate(rows):
        pair_id = f"pair_{index:07d}"
        checkpoint = private / f"{pair_id}.checkpoint.json"
        if checkpoint.exists():
            pair = signed_read(checkpoint, key)
            if pair["run_id"] != plan["run_id"] or pair["source_id"] != row["sample_id"]:
                raise ValueError("Checkpoint belongs to another run/sample")
        else:
            start = time.perf_counter()
            def material(label, length):
                return pair_material(key, plan["run_id"], row["sample_id"], label, length)
            message = int.from_bytes(material("message", (args.bits + 7) // 8), "big") & ((1 << args.bits) - 1)
            bits = format(message, f"0{args.bits}b")
            nonce, ordinary_nonce = material("rotation-nonce", 16), material("ordinary-nonce", 16)
            pair = {"run_id": plan["run_id"], "source_id": row["sample_id"], "pair_id": pair_id,
                    "donor_id": row["donor_id"], "split": row["split"],
                    "chrom": row.get("chrom"), "start": row.get("start"),
                    "nonce": nonce.hex(), "ordinary_nonce": ordinary_nonce.hex()}
            try:
                dna, metadata = encode_packet(bits, model, key=key, nonce=nonce, context=row["context"],
                                               config=config, output_bases=args.length)
                encode_seconds = time.perf_counter() - start
                decoded = decode_packet(dna, metadata, model, key=key)
                if decoded != bits:
                    raise ValueError("Roundtrip mismatch; experiment stopped")
                decode_seconds = time.perf_counter() - start - encode_seconds
                ordinary = sample_dna(args.length, model, key=key, nonce=ordinary_nonce,
                                      context=row["context"], config=config)
                pair.update({"status": "success", "metadata": metadata,
                             "sequences": {"natural": row["natural"], "ordinary": ordinary, "rrc": dna},
                             "metrics": {"pair_id": pair_id, "donor_id": row["donor_id"],
                                         "split": row["split"], **metadata["metrics"],
                                         "output_bases": len(dna), "padding_bases": len(dna) - metadata["metrics"]["encoded_bases"],
                                         "net_bits_per_base": args.bits / len(dna), "exact": True,
                                         "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                                         "total_seconds": time.perf_counter() - start}})
            except RRCBudgetError as exc:
                pair.update({"status": "budget_failure", "error": str(exc)})
            except Exception as exc:
                atomic_json(out / "run.json", {**request, "status": "fatal_error", "planned": len(rows),
                                               "sample_id": row["sample_id"], "error": str(exc)})
                raise
            signed_write(checkpoint, pair, key)
        if pair["status"] == "success":
            metrics.append(pair["metrics"])
            for kind, dna in pair["sequences"].items():
                results.append({"sample_id": pair_id + "_" + kind, "pair_id": pair_id,
                                "donor_id": row["donor_id"], "split": row["split"], "kind": kind, "sequence": dna})
            atomic_json(private / f"{pair_id}.json", pair["metadata"])
        else:
            failures.append({"sample_id": row["sample_id"], "pair_id": pair_id,
                             "donor_id": row["donor_id"], "split": row["split"], "error": pair["error"]})
        print(json.dumps({"pair": index + 1, "total": len(rows), "status": pair["status"]}), flush=True)
    if results:
        try:
            audit_carriers(results)
        except ValueError as exc:
            atomic_json(out / "run.json", {**request, "status": "fatal_error", "error": str(exc)})
            raise
        write_csv(out / "sequences.csv", results)
    atomic_json(out / "metrics.json", metrics)
    atomic_json(out / "failures.json", failures)
    atomic_json(out / "run.json", {**request, "audit": audit, "run_id": plan["run_id"],
               "status": "complete" if not failures else "complete_with_failures",
               "attempted": len(rows), "successful": len(metrics), "failed": len(failures),
               "failure_rate": len(failures) / len(rows),
               "failed_by_split": dict(Counter(r["split"] for r in failures)),
               "warning": "Stopping/padding may change distributions. Excluded budget failures cause selection bias; detector results are conditional on success."})
    if failures:
        raise RRCBudgetError(f"{len(failures)} failed pairs, recorded without redrawing. Evaluation requires explicit --allow-failures")


def verify_experiment(args):
    from .rrc_cli import read_csv
    model, key = probability_source(args), Path(args.key_file).read_bytes()
    root = Path(args.dataset)
    plan = signed_read(root / "receiver_private/plan.json", key)
    context_rows = read_csv(args.contexts)
    if hashlib.sha256(Path(args.contexts).read_bytes()).hexdigest() != plan["request"]["contexts_sha256"]:
        raise ValueError("Verification contexts differ from generation")
    by_id = {r["sample_id"]: r for r in context_rows}
    public_rows = read_csv(root / "sequences.csv")
    audit_carriers(public_rows)
    public = {r["sample_id"]: r for r in public_rows}
    successes, failed = 0, 0
    for path in sorted((root / "receiver_private").glob("*.checkpoint.json")):
        pair = signed_read(path, key)
        if pair["run_id"] != plan["run_id"]:
            raise ValueError("Foreign checkpoint")
        if pair["status"] != "success":
            failed += 1
            continue
        row = by_id[pair["source_id"]]
        bits = decode_packet(pair["sequences"]["rrc"], pair["metadata"], model, key=key)
        expected = int.from_bytes(pair_material(key, plan["run_id"], row["sample_id"], "message", (len(bits)+7)//8), "big") & ((1 << len(bits))-1)
        if int(bits, 2) != expected:
            raise ValueError("Message mismatch")
        config = RRCConfig(**plan["request"]["config"])
        ordinary = sample_dna(plan["request"]["length"], model, key=key,
                              nonce=bytes.fromhex(pair["ordinary_nonce"]), context=row["context"], config=config)
        if ordinary != pair["sequences"]["ordinary"]:
            raise ValueError("Ordinary sampling replay differs")
        for kind, sequence in pair["sequences"].items():
            expected_row = {"sample_id": pair["pair_id"] + "_" + kind, "sequence": sequence,
                            "pair_id": pair["pair_id"], "kind": kind,
                            "donor_id": row["donor_id"], "split": row["split"]}
            if public[expected_row["sample_id"]] != expected_row:
                raise ValueError("Exported detector sequence or labels changed")
        successes += 1
    if successes + failed != len(context_rows) or len(public) != 3 * successes:
        raise ValueError("Incomplete dataset/checkpoints")
    result = {"verified_pairs": successes, "recorded_failures": failed, "exact": True,
              "model": model.metadata()}
    atomic_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))
