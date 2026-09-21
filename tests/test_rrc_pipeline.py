"""CLI integration; all generated data here is synthetic test-only data."""
import csv
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RRCPipelineTest(unittest.TestCase):
    def test_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = random.Random(120)
            manifest = root / "donors.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["donor_id", "split", "fasta"])
                for split in ("train", "val", "test"):
                    fasta = root / (split + ".fa")
                    fasta.write_text(">chr1\n" + "".join(rng.choices("ACGT", k=1200)) + "\n")
                    writer.writerow([split, split, fasta])

            def run(*args):
                return subprocess.run([sys.executable, "-m", "genome_mining.rrc_cli", *map(str, args)],
                                      text=True, capture_output=True, check=True)

            contexts, model, key = root / "contexts.csv", root / "model.json", root / "key.bin"
            run("prepare-contexts", "--manifest", manifest, "--out", contexts,
                "--per-donor", 5, "--length", 64, "--context-bases", 32)
            run("fit-context-model", "--contexts", contexts, "--out", model, "--order", 2)
            run("keygen", "--out", key)
            run("encode", "--model", model, "--key-file", key, "--bits", "00101101",
                "--length", 64, "--out", root / "encoded")
            run("decode", "--model", model, "--key-file", key,
                "--dna", root / "encoded/carrier.dna.txt", "--metadata", root / "encoded/receiver.json",
                "--out", root / "decoded")
            self.assertEqual((root / "decoded/message.bits").read_text().strip(), "00101101")
            run("build", "--model", model, "--key-file", key, "--contexts", contexts,
                "--bits", 16, "--length", 64, "--context-bases", 32, "--out", root / "dataset")
            with (root / "dataset/sequences.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 45)
            self.assertEqual({len(row["sequence"]) for row in rows}, {64})
            self.assertFalse(json.loads((root / "dataset/failures.json").read_text()))
            self.assertNotIn("nonce", rows[0])
            if importlib.util.find_spec("sklearn"):
                run("evaluate", "--csv", root / "dataset/sequences.csv", "--out", root / "evaluation")
                report = json.loads((root / "evaluation/report.json").read_text())
                self.assertEqual(len(report), 3)
                for task in report.values():
                    self.assertTrue(0 <= task["auroc"] <= 1)


if __name__ == "__main__":
    unittest.main()
