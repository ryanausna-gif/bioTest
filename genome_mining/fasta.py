from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .records import SequenceWindow

DNA_ALPHABET = set("ACGTN")
STRICT_BASES = set("ACGT")


@dataclass(frozen=True)
class FastaRecord:
    name: str
    sequence: str
    description: str = ""


def clean_dna(sequence: str) -> str:
    """Normalize FASTA sequence text while keeping N as an uncertainty marker."""
    return "".join(base if base in DNA_ALPHABET else "N" for base in sequence.upper())


def acgt_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(1 for base in sequence.upper() if base in STRICT_BASES) / len(sequence)


def iter_fasta_records(path: str | Path, min_length: int = 1) -> Iterator[FastaRecord]:
    fasta_path = Path(path)
    name: str | None = None
    description = ""
    chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequence = clean_dna("".join(chunks))
                    if len(sequence) >= min_length:
                        yield FastaRecord(name=name, sequence=sequence, description=description)
                description = line[1:].strip()
                name = description.split()[0] if description else "unnamed"
                chunks = []
                continue
            chunks.append(line)

    if name is not None:
        sequence = clean_dna("".join(chunks))
        if len(sequence) >= min_length:
            yield FastaRecord(name=name, sequence=sequence, description=description)


def iter_fasta_windows(
    path: str | Path,
    window_size: int,
    step: int,
    *,
    min_acgt_fraction: float = 0.85,
    source: str = "",
    species: str = "",
    max_windows: int | None = None,
) -> Iterator[SequenceWindow]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    emitted = 0
    for record in iter_fasta_records(path, min_length=window_size):
        sequence = record.sequence
        last_start = len(sequence) - window_size
        for start in range(0, last_start + 1, step):
            window_seq = sequence[start : start + window_size]
            if acgt_fraction(window_seq) < min_acgt_fraction:
                continue
            yield SequenceWindow(
                chrom=record.name,
                start=start,
                end=start + window_size,
                sequence=window_seq,
                source=source,
                species=species,
            )
            emitted += 1
            if max_windows is not None and emitted >= max_windows:
                return
