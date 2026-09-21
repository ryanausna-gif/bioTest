from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from genome_mining.carrier_controls import RealisticCarrierGenerator
from genome_mining.candidates import extract_candidate_sequences
from genome_mining.cli import main as mining_main
from genome_mining.fasta import iter_fasta_records, iter_fasta_windows
from genome_mining.features import compute_sequence_features, compute_window_features
from genome_mining.model_inference import _local_to_genome
from genome_mining.records import SequenceWindow
from genome_mining.scoring import score_feature_rows


class GenomeMiningTest(unittest.TestCase):
    def test_reads_fasta_records_and_windows_with_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "mini.fa"
            fasta.write_text(">chr1 first chromosome\n" + "ACGT" * 20 + "\n>chr2\n" + "NNNNACGT" * 10 + "\n")

            records = list(iter_fasta_records(fasta))
            self.assertEqual([record.name for record in records], ["chr1", "chr2"])

            windows = list(iter_fasta_windows(fasta, window_size=20, step=10, min_acgt_fraction=0.9))
            self.assertGreater(len(windows), 0)
            self.assertEqual(windows[0].chrom, "chr1")
            self.assertEqual(windows[0].start, 0)
            self.assertEqual(windows[0].end, 20)
            self.assertEqual(windows[0].window_id, "chr1:0-20:+")

    def test_sequence_features_are_deterministic(self) -> None:
        features = compute_sequence_features("ACGT" * 10)
        self.assertEqual(features["length"], 40)
        self.assertAlmostEqual(features["gc_content"], 0.5)
        self.assertAlmostEqual(features["entropy_norm"], 1.0)
        self.assertEqual(features["longest_homopolymer"], 1)

    def test_scoring_ranks_low_complexity_window_first(self) -> None:
        normal = "ACGTTGCAGTCAGTCCGATG" * 4
        rows = [
            compute_window_features(SequenceWindow("chr1", idx * 80, idx * 80 + 80, normal))
            for idx in range(8)
        ]
        suspicious = SequenceWindow("chr1", 800, 880, "A" * 80)
        rows.append(compute_window_features(suspicious))

        scored, baseline = score_feature_rows(rows)

        self.assertEqual(scored[0]["window_id"], suspicious.window_id)
        self.assertGreater(float(scored[0]["anomaly_score"]), 1.0)
        self.assertIn("entropy_low", str(scored[0]["evidence"]))
        self.assertIn("gc_content", baseline)

    def test_scan_cli_writes_report_csv_bed_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "scan.fa"
            seq = ("ACGTTGCAGTCAGTCCGATG" * 12) + ("A" * 120) + ("GATCCGTAGCTAACGTTCGA" * 12)
            fasta.write_text(">chrTest\n" + seq + "\n", encoding="utf-8")
            out = root / "scan_out"

            mining_main(
                [
                    "scan",
                    "--fasta",
                    str(fasta),
                    "--out",
                    str(out),
                    "--window",
                    "80",
                    "--step",
                    "40",
                    "--threshold",
                    "0.5",
                    "--top-n",
                    "5",
                ]
            )

            self.assertTrue((out / "windows.features.csv").exists())
            self.assertTrue((out / "candidates.csv").exists())
            self.assertTrue((out / "candidates.bed").exists())
            self.assertTrue((out / "mining_report.md").exists())

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertGreater(summary["total_windows"], 0)
            self.assertGreater(summary["candidate_count"], 0)

            with (out / "candidates.csv").open("r", encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle))
            self.assertLessEqual(len(candidates), 5)
            self.assertGreater(float(candidates[0]["anomaly_score"]), 0.5)

    def test_v12_realistic_generator_can_emit_each_codec_family(self) -> None:
        generator = RealisticCarrierGenerator(length=1000)
        background = "ACGTTGCAGTCAGTCCGATG" * 50

        for codec in generator.codec_families:
            with self.subTest(codec=codec):
                result = generator.positive(background, codec, "unit_background", strength=0.6)
                self.assertEqual(result.y, 1)
                self.assertEqual(len(result.sequence), len(background))
                self.assertEqual(result.codec_family, codec)
                self.assertGreaterEqual(result.start, 0)
                self.assertGreater(result.end, result.start)

    def test_build_controls_cli_writes_v12_benchmark_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "bg.fa"
            fasta.write_text(">chrTiny\n" + ("ACGTTGCAGTCAGTCCGATG" * 80) + "\n", encoding="utf-8")
            out = root / "controls.csv"

            mining_main(
                [
                    "build-controls",
                    "--out",
                    str(out),
                    "--background-fasta",
                    str(fasta),
                    "--generator-kind",
                    "realistic",
                    "--length",
                    "500",
                    "--n-train",
                    "8",
                    "--n-val",
                    "4",
                    "--n-test",
                    "8",
                    "--negative-rate",
                    "0.25",
                    "--holdout-codecs",
                    "CodonStegoCodec",
                    "RealisticAdaptiveStegoCodec",
                    "--seed",
                    "11",
                ]
            )

            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix(".manifest.json").exists())
            with out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertIn("sequence", rows[0])
            self.assertIn("codec_family", rows[0])
            self.assertIn("open_target", rows[0])
            self.assertTrue(any(row["split"] == "test" for row in rows))
            self.assertTrue(any(row["open_target"] == "unknown_anomaly" for row in rows))

            manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["generator_kind"], "realistic")
            self.assertIn("RealisticAdaptiveStegoCodec", manifest["holdout_codecs"])
            self.assertIn("StegoCodec", manifest["calibration_codecs"])
            self.assertTrue(
                any(row["split"] == "val" and row["open_target"] == "unknown_anomaly" for row in rows)
            )
            self.assertTrue(all(row["sample_id"] for row in rows))
            self.assertTrue(set(manifest["train_codecs"]).isdisjoint(manifest["calibration_codecs"]))
            self.assertTrue(set(manifest["train_codecs"]).isdisjoint(manifest["holdout_codecs"]))

    def test_extract_candidates_from_bed_with_target_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "ref.fa"
            fasta.write_text(">chr1\n" + "ACGT" * 50 + "\n", encoding="utf-8")
            bed = root / "candidates.bed"
            bed.write_text("chr1\t10\t30\thit1\t900\t+\nchr1\t40\t60\thit2\t800\t-\n", encoding="utf-8")
            out_csv = root / "candidate_sequences.csv"
            out_fasta = root / "candidate_sequences.fa"

            rows = extract_candidate_sequences(
                fasta,
                bed,
                out_csv,
                out_fasta=out_fasta,
                target_length=40,
                min_acgt_fraction=0.9,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["candidate_id"], "hit1")
            self.assertEqual(rows[0]["sequence_length"], 40)
            self.assertEqual(len(rows[1]["sequence"]), 40)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_fasta.exists())
            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(exported[0]["chrom"], "chr1")
            self.assertIn("sequence", exported[0])

    def test_negative_strand_localization_maps_back_to_genome(self) -> None:
        row = {
            "strand": "-",
            "extract_start": 100,
            "extract_end": 140,
            "left_pad": 0,
            "right_pad": 0,
        }
        self.assertEqual(_local_to_genome(row, 5, 15), (125, 135))

    def test_v12_model_cli_commands_are_registered(self) -> None:
        for command in [
            "train-baseline",
            "train-pe-nac-plus",
            "train-ef-pe-nac-plus",
            "train-deep-carrier",
            "evaluate-openood",
            "run-ood-suite",
            "train-evidence-fusion",
            "calibrate-unknown",
            "analyze-benchmark",
            "make-benchmark-card",
            "analyze-failures",
            "evaluate-localization",
            "plot-embeddings",
            "make-leaderboard",
            "compare-predictions",
            "run-encoder-comparison",
            "run-ablations",
            "run-holdout-matrix",
            "run-v12-suite",
            "extract-candidates",
            "score-candidates",
            "build-image-dataset",
            "audit-image-dataset",
            "train-image-locator",
            "evaluate-image-locator",
        ]:
            with self.subTest(command=command):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        mining_main([command, "--help"])
                self.assertEqual(ctx.exception.code, 0)

    def test_v12_suite_dry_run_uses_unified_module_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mining_main(
                    [
                        "run-v12-suite",
                        "--out-root",
                        str(Path(tmp) / "suite"),
                        "--n-train",
                        "8",
                        "--n-val",
                        "8",
                        "--n-test",
                        "8",
                        "--length",
                        "128",
                        "--epochs",
                        "1",
                        "--encoders",
                        "cnn",
                        "hybrid",
                        "--dry-run",
                    ]
                )
            rendered = output.getvalue()
            self.assertIn("-m genome_mining build-controls", rendered)
            self.assertIn("-m genome_mining train-pe-nac-plus", rendered)
            self.assertIn("-m genome_mining train-ef-pe-nac-plus", rendered)
            self.assertIn("-m genome_mining train-deep-carrier", rendered)
            self.assertIn("-m genome_mining calibrate-unknown", rendered)
            self.assertIn("-m genome_mining run-ood-suite", rendered)
            self.assertIn("validation_predictions.csv", rendered)
            self.assertNotIn("scripts/train_carrier_baseline.py", rendered)


if __name__ == "__main__":
    unittest.main()
