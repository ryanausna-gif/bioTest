from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from genome_mining.image_dataset import audit_image_genome_dataset, build_image_genome_dataset
from genome_mining.image_payload import (
    RawImageDNASpec,
    bytes_to_dna,
    compare_pixel_bytes,
    decode_image_dna,
    dna_to_bytes,
    encode_image_file,
)
from genome_mining.stego_validation import verify_stego_roundtrip_dataset


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class ImagePayloadTest(unittest.TestCase):
    def test_raw_grayscale_image_round_trip_is_exact(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gradient.png"
            pixels = bytes(range(16))
            Image.frombytes("L", (4, 4), pixels).save(path)
            spec = RawImageDNASpec(width=4, height=4)

            dna, encoded_pixels = encode_image_file(path, spec)
            recovered = dna_to_bytes(dna, expected_bytes=16)
            image = decode_image_dna(dna, spec)

            self.assertEqual(len(dna), 64)
            self.assertEqual(encoded_pixels, recovered)
            self.assertEqual(image.tobytes(), recovered)
            self.assertTrue(compare_pixel_bytes(encoded_pixels, recovered)["exact"])

    def test_two_bit_mapping_has_all_byte_values(self) -> None:
        payload = bytes(range(256))
        dna = bytes_to_dna(payload)
        self.assertEqual(len(dna), 1024)
        self.assertEqual(dna_to_bytes(dna), payload)


@unittest.skipUnless(HAS_NUMPY, "numpy is required for the memory-mapped dataset smoke test")
class ImageGenomeDatasetTest(unittest.TestCase):
    def test_build_and_audit_disjoint_dataset(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome_rows = []
            patterns = {"train": "ACGT", "val": "AGCT", "test": "ATGC"}
            for split, count in (("train", 4), ("val", 2), ("test", 2)):
                fasta = root / f"{split}.fa"
                fasta.write_text(
                    ">chr1\n" + (patterns[split] * (count * 32 + 8)) + "\n",
                    encoding="utf-8",
                )
                genome_rows.append(
                    {
                        "donor_id": f"donor_{split}",
                        "assembly_id": f"assembly_{split}",
                        "fasta": str(fasta),
                        "split": split,
                    }
                )
            genome_manifest = root / "genomes.csv"
            with genome_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(genome_rows[0]))
                writer.writeheader()
                writer.writerows(genome_rows)

            image_root = root / "images"
            image_root.mkdir()
            for index in range(4):
                Image.new("L", (8, 6), color=20 + index * 40).save(image_root / f"image_{index}.png")

            dataset = root / "dataset"
            manifest = build_image_genome_dataset(
                genome_manifest,
                dataset,
                image_root=image_root,
                n_train=4,
                n_val=2,
                n_test=2,
                positive_fraction=0.5,
                image_size=4,
                window_length=128,
                min_acgt_fraction=1.0,
                seed=7,
            )
            report = audit_image_genome_dataset(dataset)
            roundtrip = verify_stego_roundtrip_dataset(
                dataset,
                root / "roundtrip",
            )

            self.assertEqual(manifest["image_spec"]["payload_bases"], 64)
            self.assertEqual(report["n_samples"], 8)
            self.assertEqual(report["positive_counts"], {"train": 2, "val": 1, "test": 1})
            self.assertTrue(report["passed"])
            self.assertEqual(report["pixel_hash_leakage"], [])
            self.assertEqual(report["reused_pixel_hashes"], [])
            self.assertTrue(roundtrip["passed"])
            self.assertEqual(roundtrip["n_checked"], 4)

    def test_genome_manifest_rejects_donor_split_leakage(self) -> None:
        from genome_mining.image_dataset import read_genome_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta_a = root / "a.fa"
            fasta_b = root / "b.fa"
            fasta_a.write_text(">chr1\nACGT\n", encoding="utf-8")
            fasta_b.write_text(">chr1\nACGT\n", encoding="utf-8")
            manifest = root / "genomes.csv"
            manifest.write_text(
                "donor_id,assembly_id,fasta,split\n"
                f"same,a,{fasta_a},train\n"
                f"same,b,{fasta_b},test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_genome_manifest(manifest)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for the locator forward smoke test")
class ImageLocatorTest(unittest.TestCase):
    def test_model_forward_loss_and_decode_shapes(self) -> None:
        import torch

        from genome_mining.models.image_locator import (
            HierarchicalImageLocator,
            decode_locator_output,
            image_locator_loss,
        )

        model = HierarchicalImageLocator(
            d_model=32,
            downsample_stride=16,
            context="dilated_cnn",
            context_layers=1,
            dropout=0.0,
        )
        tokens = torch.randint(0, 5, (2, 128))
        output = model(tokens)
        labels = torch.tensor([1.0, 0.0])
        starts = torch.tensor([37, 0])
        losses = image_locator_loss(output, labels, starts, payload_bases=64, stride=16)
        decoded = decode_locator_output(output, payload_bases=64, stride=16, window_length=128)

        self.assertEqual(tuple(output["start_bin_logits"].shape), (2, 8))
        self.assertEqual(tuple(output["offset_logits"].shape), (2, 8, 16))
        self.assertEqual(tuple(decoded["start"].shape), (2,))
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
