"""Sampling and grouped audits for continuous-carrier experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
import time
from pathlib import Path


def dna_signature(sequence):
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return hashlib.sha256(min(sequence, reverse).encode()).hexdigest()


def audit_contexts(rows, length=None):
    if not rows:
        raise ValueError("Empty contexts dataset")
    seen, donors, sequences = set(), {}, {}
    intervals = defaultdict(list)
    for row in rows:
        sid, split, donor = row["sample_id"], row["split"], row["donor_id"]
        if not sid or not donor or sid in seen or split not in {"train", "val", "test"}:
            raise ValueError("Duplicate ID, empty donor or invalid split")
        seen.add(sid)
        if donor in donors and donors[donor] != split:
            raise ValueError("Donor leakage across splits")
        donors[donor] = split
        for name in ("context", "natural"):
            sequence = row[name]
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError("Use nonempty uppercase ACGT contexts and natural sequences")
            signature = dna_signature(sequence)
            if signature in sequences and sequences[signature] != split:
                raise ValueError("Exact/reverse-complement sequence leakage across splits")
            sequences[signature] = split
        if length is not None and len(row["natural"]) != length:
            raise ValueError("Natural length must equal fixed output length")
        if "start" in row and "chrom" in row:
            start = int(row["start"])
            if start < 0 or not row["chrom"]:
                raise ValueError("Invalid genomic coordinates")
            intervals[(donor, row["chrom"])].append((start, start + len(row["context"]) + len(row["natural"])))
    for locus, spans in intervals.items():
        previous_end = -1
        for start, end in sorted(spans):
            if start < previous_end:
                raise ValueError(f"Overlapping source windows at {locus}")
            previous_end = end
    return {"rows": len(rows), "donors": dict(Counter(donors.values())),
            "rows_by_split": dict(Counter(r["split"] for r in rows)),
            "near_homology_checked": False}


def select_windows(windows, count, seed, mode, progress=None):
    """Uniform reservoir over eligible nonoverlapping grid windows, not all starts."""
    rng, selected, eligible = random.Random(seed), [], 0
    updated = time.monotonic()
    for inspected, (chrom, start, sequence) in enumerate(windows, 1):
        if progress and time.monotonic() - updated > 30:
            progress(inspected, eligible)
            updated = time.monotonic()
        if set(sequence) - set("ACGT"):
            continue
        eligible += 1
        if len(selected) < count:
            selected.append((chrom, start, sequence))
        elif mode == "reservoir":
            slot = rng.randrange(eligible)
            if slot < count:
                selected[slot] = (chrom, start, sequence)
        if mode == "first" and len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} clean windows, found only {eligible}")
    return sorted(selected, key=lambda x: (x[0], x[1])), eligible


def prepare_contexts(args):
    from .rrc_cli import context_records, read_csv, write_csv, write_json
    if args.per_donor < 1 or args.length < 1 or args.context_bases < 1:
        raise ValueError("Sizes must be positive")
    if Path(args.out).exists() or Path(str(args.out) + ".audit.json").exists():
        raise ValueError("Output already exists")
    manifest = read_csv(args.manifest)
    donors, files, rows, sources = set(), set(), [], []
    for item in manifest:
        donor, split = item["donor_id"], item["split"]
        fasta = Path(item["fasta"])
        if not fasta.is_absolute():
            fasta = Path(args.manifest).resolve().parent / fasta
        fasta = fasta.resolve()
        if not donor or donor in donors or str(fasta) in files or split not in {"train", "val", "test"}:
            raise ValueError("Unique donors/files and explicit splits required")
        donors.add(donor)
        files.add(str(fasta))
        donor_seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{donor}".encode()).digest()[:8], "big")
        print(f"Scanning {donor} ({args.sampling}): {fasta}", flush=True)
        selected, eligible = select_windows(context_records(fasta, args.context_bases + args.length),
            args.per_donor, donor_seed, args.sampling,
            progress=lambda inspected, eligible: print(
                f"{donor}: inspected={inspected}, eligible={eligible}", flush=True))
        for chrom, start, sequence in selected:
            rows.append({"sample_id": f"sample_{len(rows):07d}", "donor_id": donor,
                         "split": split, "chrom": chrom, "start": start,
                         "context": sequence[:args.context_bases], "natural": sequence[args.context_bases:]})
        sources.append({"donor_id": donor, "path": str(fasta), "eligible_windows": eligible,
                        "file_bytes": fasta.stat().st_size, "mtime_ns": fasta.stat().st_mtime_ns})
        print(f"Prepared {donor}: {len(selected)} of {eligible} clean grid windows", flush=True)
    audit = audit_contexts(rows, args.length)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    write_json(str(args.out) + ".audit.json", {**audit, "sampling": args.sampling,
               "seed": args.seed, "sources": sources,
               "contexts_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest()})


def audit_carriers(rows):
    if not rows:
        raise ValueError("Empty carrier dataset")
    seen, donors, pairs, signatures = set(), {}, {}, {}
    for row in rows:
        if row["sample_id"] in seen or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Duplicate sample ID or invalid split")
        seen.add(row["sample_id"])
        if not row["sequence"] or set(row["sequence"]) - set("ACGT"):
            raise ValueError("Invalid carrier DNA")
        donor, split, pair = row["donor_id"], row["split"], row["pair_id"]
        if not donor or not pair or (donor in donors and donors[donor] != split):
            raise ValueError("Empty group or donor leakage")
        donors[donor] = split
        entry = pairs.setdefault(pair, {"donor": donor, "split": split, "kinds": set(), "lengths": set()})
        if entry["donor"] != donor or entry["split"] != split or row["kind"] in entry["kinds"]:
            raise ValueError("Invalid paired groups or pair leakage")
        entry["kinds"].add(row["kind"])
        entry["lengths"].add(len(row["sequence"]))
        sig = dna_signature(row["sequence"])
        if sig in signatures and signatures[sig] != split:
            raise ValueError("Exact/RC carrier sequence leakage")
        signatures[sig] = split
    for entry in pairs.values():
        if entry["kinds"] != {"natural", "ordinary", "rrc"} or len(entry["lengths"]) != 1:
            raise ValueError("Each pair needs three equal-length kinds")
    if len({len(r["sequence"]) for r in rows}) != 1:
        raise ValueError("Use a single fixed length per experiment")
    return {"pairs": len(pairs), "donors_by_split": dict(Counter(donors.values())),
            "exact_rc_checked": True, "near_homology_checked": False}
