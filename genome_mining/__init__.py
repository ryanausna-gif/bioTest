"""Coordinate-aware genome mining utilities."""

from .fasta import FastaRecord, iter_fasta_records, iter_fasta_windows
from .features import compute_window_features
from .records import MiningHit, SequenceWindow
from .scoring import score_feature_rows
from .carrier_controls import RealisticCarrierGenerator, build_carrier_benchmark

__all__ = [
    "FastaRecord",
    "MiningHit",
    "SequenceWindow",
    "compute_window_features",
    "iter_fasta_records",
    "iter_fasta_windows",
    "score_feature_rows",
    "RealisticCarrierGenerator",
    "build_carrier_benchmark",
]
