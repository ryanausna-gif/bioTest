import csv
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from genome_mining.rrc_data import select_windows, audit_contexts, audit_carriers
from genome_mining.rrc_experiment import signed_write, signed_read, pair_material, build_experiment
from genome_mining.lm_stego.model_adapter import KmerProbabilityModel


class RRCExperimentTests(unittest.TestCase):
    def test_reservoir_reproducible_and_not_first(self):
        rows = [("c", i*4, "ACGT") for i in range(100)]
        a, n = select_windows(iter(rows), 10, 2026, "reservoir")
        b, _ = select_windows(iter(rows), 10, 2026, "reservoir")
        self.assertEqual(a, b)
        self.assertEqual(n, 100)
        self.assertTrue(any(start >= 40 for _, start, _ in a))
        first, n = select_windows(iter(rows), 10, 2026, "first")
        self.assertEqual(first, rows[:10])
        self.assertEqual(n, 10)

    def test_overlap_and_pairs(self):
        rows = [{"sample_id": "a", "donor_id": "d", "split": "train", "chrom": "c",
                 "start": 0, "context": "AC", "natural": "AA"},
                {"sample_id": "b", "donor_id": "d", "split": "train", "chrom": "c",
                 "start": 3, "context": "GT", "natural": "CC"}]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            audit_contexts(rows)
        carriers = [{"sample_id": "a", "pair_id": "p", "donor_id": "d", "split": "train",
                     "kind": "rrc", "sequence": "ACGT"}]
        with self.assertRaisesRegex(ValueError, "three"):
            audit_carriers(carriers)

    def test_checkpoint_tamper_and_domain_separation(self):
        key = bytes(range(32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            signed_write(path, {"value": 1}, key)
            self.assertEqual(signed_read(path, key), {"value": 1})
            changed = json.loads(path.read_text())
            changed["data"]["value"] = 2
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "authentication"):
                signed_read(path, key)
        self.assertNotEqual(pair_material(key, "run", "id", "ordinary", 16),
                            pair_material(key, "run", "id", "rotation", 16))

    def test_resume_and_budget_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / "key"
            key.write_bytes(bytes(range(32)))
            model = root / "model.json"
            KmerProbabilityModel(order=0).save(model)
            contexts = root / "contexts.csv"
            contexts.write_text("sample_id,donor_id,split,context,natural\na,d,train,ACGT,ACGTACGTACGTACGT\n")
            args = SimpleNamespace(contexts=str(contexts), key_file=str(key), model=str(model), hf_model=None,
                                   bits=8, length=16, max_bases=64, precision_bits=12, context_bases=8,
                                   device="cpu", out=str(root/"run"), resume=False)
            build_experiment(args)
            before = (root/"run/sequences.csv").read_bytes()
            args.resume = True
            build_experiment(args)
            self.assertEqual((root/"run/sequences.csv").read_bytes(), before)
            args.bits = 9
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                build_experiment(args)
            args.out, args.resume, args.bits = str(root/"failed"), False, 2048
            with self.assertRaises(ValueError):
                build_experiment(args)
            checkpoint = (root/"failed/receiver_private/pair_0000000.checkpoint.json").read_bytes()
            args.resume = True
            with self.assertRaises(ValueError):
                build_experiment(args)
            self.assertEqual((root/"failed/receiver_private/pair_0000000.checkpoint.json").read_bytes(), checkpoint)
            run = json.loads((root/"failed/run.json").read_text())
            self.assertEqual(run["failure_rate"], 1)

    @unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("torch"), "optional detector dependencies")
    def test_saved_cnn_lr_and_verification(self):
        from genome_mining.rrc_cli import write_csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = random.Random(90)
            rows = []
            for split in ("train", "val", "test"):
                for donor in range(2):
                    for index in range(3):
                        rows.append({"sample_id": f"{split}{donor}{index}", "donor_id": f"{split}{donor}",
                                     "split": split, "context": "".join(rng.choices("ACGT", k=16)),
                                     "natural": "".join(rng.choices("ACGT", k=32))})
            write_csv(root/"contexts.csv", rows)
            KmerProbabilityModel(order=0).save(root/"model.json")
            (root/"key").write_bytes(bytes(range(32)))
            def run(*args):
                completed = subprocess.run([sys.executable, "-m", "genome_mining.rrc_cli", *map(str, args)],
                                           capture_output=True, text=True, timeout=180)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            common = ["--model", root/"model.json", "--key-file", root/"key", "--context-bases", 16]
            run("build", *common, "--contexts", root/"contexts.csv", "--bits", 8,
                "--length", 32, "--out", root/"data")
            run("verify", *common, "--contexts", root/"contexts.csv", "--dataset", root/"data")
            for detector in ("logreg", "cnn"):
                run("evaluate", "--csv", root/"data/sequences.csv", "--detector", detector,
                    "--epochs", 2, "--bootstrap", 10, "--out", root/detector)
                run("score", "--detector-dir", root/detector/"ordinary_vs_rrc",
                    "--csv", root/"data/sequences.csv", "--out", root/(detector + "_scores.csv"))
                from genome_mining.rrc_cli import read_csv
                saved = {r["sample_id"]: float(r["score"]) for r in read_csv(root/(detector + "_scores.csv"))}
                for row in read_csv(root/detector/"predictions.csv"):
                    if row["task"] == "ordinary_vs_rrc":
                        self.assertAlmostEqual(saved[row["sample_id"]], float(row["score"]), places=6)
                report = json.loads((root/detector/"report.json").read_text())
                self.assertIsNotNone(report["ordinary_vs_rrc"]["donor_bootstrap_95ci"])
                run("evaluate-transfer", "--detector-dir", root/detector/"ordinary_vs_rrc",
                    "--csv", root/"data/sequences.csv", "--bootstrap", 10, "--out", root/(detector+"_transfer"))
                transfer = json.loads((root/(detector+"_transfer")/"report.json").read_text())
                self.assertEqual(transfer["auroc"], report["ordinary_vs_rrc"]["auroc"])
                self.assertFalse(transfer["recalibrated"])
                metadata_path = root/detector/"ordinary_vs_rrc/detector.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["fit_donors"].append("test0")
                metadata_path.write_text(json.dumps(metadata))
                result = subprocess.run([sys.executable, "-m", "genome_mining.rrc_cli", "evaluate-transfer",
                    "--detector-dir", str(metadata_path.parent), "--csv", str(root/"data/sequences.csv"),
                    "--out", str(root/(detector+"_leak"))], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("overlap", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional detector dependencies")
    def test_suite_resume(self):
        from genome_mining.rrc_cli import write_csv
        from genome_mining.rrc_suite import resolve_config
        with tempfile.TemporaryDirectory() as tmp:
            root, rng = Path(tmp), random.Random(123)
            rows = [{"sample_id": f"{split}{i}", "donor_id": split, "split": split,
                     "context": "".join(rng.choices("ACGT", k=16)),
                     "natural": "".join(rng.choices("ACGT", k=32))}
                    for split in ("train", "val", "test") for i in range(3)]
            write_csv(root/"contexts.csv", rows)
            KmerProbabilityModel(order=0).save(root/"model.json")
            (root/"key").write_bytes(bytes(range(32)))
            config = {"contexts": "contexts.csv", "key_file": "key", "model": "model.json",
                      "out": "suite", "bits": [8, 16], "length": 32, "context_bases": 16,
                      "detectors": ["logreg"], "bootstrap": 0}
            (root/"config.json").write_text(json.dumps(config))
            resolved = resolve_config(root/"config.json")
            self.assertEqual(resolved["contexts"], str(root/"contexts.csv"))
            for flags in ([], ["--resume"]):
                result = subprocess.run([sys.executable, "-m", "genome_mining.rrc_suite",
                    "--config", str(root/"config.json"), *flags], capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            from genome_mining.rrc_cli import read_csv
            self.assertEqual(len(read_csv(root/"suite/summary.csv")), 6)
            self.assertFalse((root/"suite/bits_8/logreg_attempt_2").exists())

    @unittest.skipUnless(os.environ.get("RRC_TEST_HF_MODEL"), "set RRC_TEST_HF_MODEL to a trusted local checkpoint")
    def test_real_hf_build_verify_and_frozen_probe(self):
        from genome_mining.rrc_cli import write_csv, read_csv
        with tempfile.TemporaryDirectory() as tmp:
            root, rng = Path(tmp), random.Random(124)
            rows = [{"sample_id": f"{split}{i}", "donor_id": split, "split": split,
                     "context": "".join(rng.choices("ACGT", k=16)),
                     "natural": "".join(rng.choices("ACGT", k=32))}
                    for split in ("train", "val", "test") for i in range(2)]
            write_csv(root/"contexts.csv", rows)
            (root/"key").write_bytes(bytes(range(32)))
            def run(*args):
                result = subprocess.run([sys.executable, "-m", "genome_mining.rrc_cli", *map(str, args)],
                    capture_output=True, text=True, timeout=240)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            hf = ["--hf-model", os.environ["RRC_TEST_HF_MODEL"], "--trust-remote-code"]
            common = [*hf, "--key-file", root/"key", "--context-bases", 16]
            run("build", *common, "--contexts", root/"contexts.csv", "--bits", 8,
                "--length", 32, "--out", root/"data")
            run("verify", *common, "--contexts", root/"contexts.csv", "--dataset", root/"data")
            run("evaluate", *hf, "--csv", root/"data/sequences.csv", "--detector", "hyena-probe",
                "--bootstrap", 0, "--out", root/"probe")
            run("score", *hf, "--detector-dir", root/"probe/ordinary_vs_rrc",
                "--csv", root/"data/sequences.csv", "--out", root/"scores.csv")
            saved = {r["sample_id"]: float(r["score"]) for r in read_csv(root/"scores.csv")}
            for row in read_csv(root/"probe/predictions.csv"):
                if row["task"] == "ordinary_vs_rrc":
                    self.assertAlmostEqual(saved[row["sample_id"]], float(row["score"]), places=6)


if __name__ == "__main__":
    unittest.main()
