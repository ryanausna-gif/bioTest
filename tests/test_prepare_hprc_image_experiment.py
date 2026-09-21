from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_hprc_image_experiment import select_assemblies, write_outputs


class PrepareHprcImageExperimentTest(unittest.TestCase):
    def test_selects_donors_before_haplotypes_and_keeps_splits_disjoint(self) -> None:
        rows = []
        for donor_index in range(8):
            donor = f"HG{donor_index:05d}"
            for haplotype in ("1", "2"):
                name = f"{donor}_hap{haplotype}"
                rows.append(
                    {
                        "sample_id": donor,
                        "haplotype": haplotype,
                        "assembly_name": name,
                        "assembly": f"s3://bucket/{name}.fa.gz",
                        "assembly_md5": "",
                    }
                )

        selected = select_assemblies(
            rows,
            n_train=4,
            n_val=2,
            n_test=2,
            haplotypes="both",
            seed=11,
        )
        donor_splits = {}
        for row in selected:
            donor_splits.setdefault(row["donor_id"], set()).add(row["split"])

        self.assertEqual(len(selected), 16)
        self.assertTrue(all(len(splits) == 1 for splits in donor_splits.values()))
        self.assertTrue(all(sum(row["donor_id"] == donor for row in selected) == 2 for donor in donor_splits))

        repeated = select_assemblies(
            rows,
            n_train=4,
            n_val=2,
            n_test=2,
            haplotypes="both",
            seed=11,
        )
        self.assertEqual(selected, repeated)

    def test_writes_download_script_and_genome_manifest(self) -> None:
        selected = [
            {
                "donor_id": "HG00001",
                "assembly_id": "HG00001_hap1",
                "haplotype": "1",
                "split": "train",
                "uri": "s3://bucket/HG00001_hap1.fa.gz",
                "filename": "HG00001_hap1.fa.gz",
                "md5_uri": "s3://bucket/HG00001_hap1.fa.gz.md5",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_outputs(selected, root / "plan", root / "fasta")
            script = (root / "plan" / "download_and_unpack_hprc.sh").read_text(encoding="utf-8")
            offline_script = (root / "plan" / "verify_and_unpack_hprc_offline.sh").read_text(
                encoding="utf-8"
            )
            windows_script = (root / "plan" / "download_hprc_windows.ps1").read_text(
                encoding="utf-8-sig"
            )
            urls = (root / "plan" / "hprc_download_urls.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            with (root / "plan" / "genome_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                manifest = list(csv.DictReader(handle))
            with (root / "plan" / "donor_split.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                donor_split = list(csv.DictReader(handle))

            self.assertIn("https://bucket.s3.amazonaws.com/HG00001_hap1.fa.gz", script)
            self.assertIn("curl -L --fail --retry 8", script)
            self.assertNotIn("aws s3", script)
            self.assertIn('|| ! "$GZIP_TOOL" -t', script)
            self.assertIn('"$GZIP_TOOL" -dk', script)
            self.assertIn("https://bucket.s3.amazonaws.com/HG00001_hap1.fa.gz", windows_script)
            self.assertIn("https://bucket.s3.amazonaws.com/HG00001_hap1.fa.gz.md5", windows_script)
            self.assertIn("Get-FileHash -LiteralPath $Target -Algorithm MD5", windows_script)
            self.assertIn("curl.exe -L --fail --retry 8", windows_script)
            self.assertIn("windows_sha256s.txt", windows_script)
            self.assertNotIn("AWS CLI", windows_script)
            self.assertIn('GZIP_TOOL=gzip', offline_script)
            self.assertIn('"$GZIP_TOOL" -t', offline_script)
            self.assertEqual(urls, ["https://bucket.s3.amazonaws.com/HG00001_hap1.fa.gz"])
            self.assertTrue(manifest[0]["fasta"].endswith("HG00001_hap1.fa"))
            self.assertEqual(manifest[0]["donor_id"], "HG00001")
            self.assertEqual(
                donor_split,
                [
                    {
                        "donor_id": "HG00001",
                        "split": "train",
                        "assembly_count": "1",
                        "haplotypes": "1",
                    }
                ],
            )

    def test_preserves_posix_server_path_when_plan_is_generated_on_windows(self) -> None:
        selected = [
            {
                "donor_id": "HG00002",
                "assembly_id": "HG00002_hap1",
                "haplotype": "1",
                "split": "test",
                "uri": "s3://bucket/HG00002_hap1.fa.gz",
                "filename": "HG00002_hap1.fa.gz",
                "md5_uri": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan"
            write_outputs(selected, out, "/public/home/user/hprc/fasta")
            with (out / "genome_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                manifest = list(csv.DictReader(handle))
            offline_script = (out / "verify_and_unpack_hprc_offline.sh").read_text(
                encoding="utf-8"
            )

            self.assertEqual(
                manifest[0]["fasta"], "/public/home/user/hprc/fasta/HG00002_hap1.fa"
            )
            self.assertIn("FASTA_DIR=/public/home/user/hprc/fasta", offline_script)


if __name__ == "__main__":
    unittest.main()
