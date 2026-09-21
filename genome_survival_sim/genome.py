from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random


@dataclass(frozen=True)
class Chromosome:
    name: str
    length: int
    recombination_rate: float


@dataclass(frozen=True)
class GenomeModel:
    species: str
    chromosomes: tuple[Chromosome, ...]
    snv_rate: float
    indel_rate: float
    notes: str

    def chromosome(self, name: str) -> Chromosome:
        for chrom in self.chromosomes:
            if chrom.name == name:
                return chrom
        raise KeyError(name)


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int
    label: str = "candidate"

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class Locus:
    chrom: str
    start: int
    end: int
    label: str = "intergenic_random"


def default_genome(species: str) -> GenomeModel:
    normalized = species.lower().replace("-", "_")
    if normalized in {"human", "human_t2t", "t2t", "chm13"}:
        return GenomeModel(
            species="human_t2t",
            chromosomes=tuple(
                Chromosome(f"chr{i}", length, 1.1e-8)
                for i, length in enumerate(
                    [
                        248_387_328,
                        242_696_752,
                        201_105_948,
                        193_574_945,
                        182_045_439,
                        172_126_628,
                        160_567_428,
                        146_259_331,
                        150_617_247,
                        134_758_134,
                        135_127_769,
                        133_324_548,
                        113_566_686,
                        101_161_492,
                        99_753_195,
                        96_330_374,
                        84_276_897,
                        80_542_538,
                        61_707_364,
                        66_210_255,
                        45_090_682,
                        51_324_926,
                    ],
                    start=1,
                )
            )
            + (Chromosome("chrX", 154_259_566, 1.1e-8),),
            snv_rate=1.2e-8,
            indel_rate=1.0e-9,
            notes="Approximate T2T-CHM13-scale autosomes/X; use --chrom-sizes for exact local references.",
        )
    if normalized in {"mouse", "mus_musculus", "grcm39", "mm39"}:
        lengths = [
            195_154_279,
            181_755_017,
            159_745_316,
            156_860_686,
            151_758_149,
            149_588_044,
            144_995_196,
            130_127_694,
            124_359_700,
            130_530_862,
            121_973_369,
            120_092_757,
            120_883_175,
            125_139_656,
            104_073_951,
            98_089_268,
            95_207_651,
            90_755_026,
            61_431_566,
        ]
        return GenomeModel(
            species="mouse_grcm39",
            chromosomes=tuple(Chromosome(f"chr{i}", length, 1.0e-8) for i, length in enumerate(lengths, start=1))
            + (Chromosome("chrX", 169_476_592, 1.0e-8),),
            snv_rate=5.4e-9,
            indel_rate=5.0e-10,
            notes="Approximate GRCm39-scale chromosomes; use --chrom-sizes for exact local references.",
        )
    if normalized in {"fly", "drosophila", "drosophila_melanogaster", "dm6"}:
        lengths = {
            "chr2L": 23_513_712,
            "chr2R": 25_286_936,
            "chr3L": 28_110_227,
            "chr3R": 32_079_331,
            "chr4": 1_348_131,
            "chrX": 23_542_271,
        }
        return GenomeModel(
            species="drosophila_dm6",
            chromosomes=tuple(Chromosome(name, length, 2.4e-8) for name, length in lengths.items()),
            snv_rate=3.5e-9,
            indel_rate=3.0e-10,
            notes="Approximate dm6 major chromosomes; use --chrom-sizes for exact local references.",
        )
    if normalized in {"toy", "test"}:
        return GenomeModel(
            species="toy",
            chromosomes=(Chromosome("chrToy", 1_000_000, 1.0e-8),),
            snv_rate=1.0e-8,
            indel_rate=1.0e-10,
            notes="Small toy genome for tests and demos.",
        )
    raise ValueError(f"Unknown species preset: {species}")


def load_chrom_sizes(path: str | Path, species: str = "custom") -> GenomeModel:
    chroms: list[Chromosome] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            chroms.append(Chromosome(parts[0], int(parts[1]), 1.0e-8))
    if not chroms:
        raise ValueError(f"No chromosome sizes found in {source}")
    return GenomeModel(
        species=species,
        chromosomes=tuple(chroms),
        snv_rate=1.0e-8,
        indel_rate=1.0e-9,
        notes=f"Loaded chromosome sizes from {source}.",
    )


def load_candidate_bed(path: str | Path) -> list[Interval]:
    intervals: list[Interval] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            label = parts[3] if len(parts) >= 4 else "candidate_bed"
            intervals.append(Interval(parts[0], int(parts[1]), int(parts[2]), label))
    if not intervals:
        raise ValueError(f"No intervals found in {source}")
    return intervals


def select_random_locus(
    genome: GenomeModel,
    rng: Random,
    length: int,
    label: str,
    intervals: list[Interval] | None = None,
) -> Locus:
    if intervals:
        viable = [interval for interval in intervals if interval.length >= length]
        if not viable:
            raise ValueError("No candidate BED intervals are large enough for the payload fragment.")
        total = sum(interval.length for interval in viable)
        pick = rng.randrange(total)
        running = 0
        for interval in viable:
            running += interval.length
            if pick < running:
                start = rng.randrange(interval.start, interval.end - length + 1)
                return Locus(interval.chrom, start, start + length, interval.label)

    total_length = sum(chrom.length for chrom in genome.chromosomes if chrom.length >= length + 2)
    if total_length <= 0:
        raise ValueError("Genome is too small for payload fragment placement.")
    pick = rng.randrange(total_length)
    running = 0
    for chrom in genome.chromosomes:
        if chrom.length < length + 2:
            continue
        running += chrom.length
        if pick < running:
            start = rng.randrange(1, chrom.length - length)
            return Locus(chrom.name, start, start + length, label)
    raise RuntimeError("Failed to select locus.")
