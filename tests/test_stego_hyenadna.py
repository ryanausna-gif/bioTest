from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

from genome_mining.stego_codecs import DirectATCGCodec, make_stego_codec
from genome_mining.stego_metrics import kmer_js_divergence, payload_stealth_metrics


HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class DirectStegoCodecTest(unittest.TestCase):
    def test_requested_atcg_mapping_and_round_trip(self) -> None:
        codec = DirectATCGCodec()
        payload = bytes([0b00011011])
        dna = codec.encode(payload)
        self.assertEqual(dna, "ATCG")
        self.assertEqual(codec.decode(dna, expected_bytes=1), payload)

    def test_stealth_metrics_are_finite(self) -> None:
        metrics = payload_stealth_metrics("ACGTACGT", "ACGTTCGT", source_bits=8, kmer_order=2)
        self.assertEqual(metrics["effective_bits_per_base"], 1.0)
        self.assertAlmostEqual(metrics["cover_change_fraction"], 0.125)
        self.assertGreaterEqual(kmer_js_divergence("AAAA", "CCCC", k=2), 0.0)


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "cryptography is required for authenticated codecs")
class AuthenticatedStegoCodecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.key = b"test-key-material-that-is-32-bytes!"
        self.payload = bytes(range(128))
        self.cover = ("ACGTTGCA" * 10000)[:70000]

    def test_encrypted_and_constrained_round_trip(self) -> None:
        for name in ("encrypted", "constrained", "kmer"):
            codec = make_stego_codec(name, key=self.key)
            dna = codec.encode(self.payload, cover=self.cover, sample_id="sample-1")
            recovered = codec.decode(dna, expected_bytes=len(self.payload), sample_id="sample-1")
            self.assertEqual(recovered, self.payload)
            self.assertLessEqual(len(dna), codec.max_encoded_bases(len(self.payload)))

    def test_constrained_words_obey_gc_and_homopolymer_limits(self) -> None:
        codec = make_stego_codec("constrained", key=self.key)
        dna = codec.encode(self.payload, sample_id="sample-2")
        for index in range(0, len(dna), 6):
            word = dna[index : index + 6]
            self.assertIn(word.count("G") + word.count("C"), (2, 3, 4))
            self.assertNotIn("AAAA", word)
            self.assertNotIn("CCCC", word)
            self.assertNotIn("GGGG", word)
            self.assertNotIn("TTTT", word)

    def test_cover_hamming_is_reversible_and_gc_preserving(self) -> None:
        codec = make_stego_codec("cover", key=self.key)
        dna = codec.encode(self.payload, cover=self.cover, sample_id="sample-3")
        source = self.cover[: len(dna)]
        changed = sum(left != right for left, right in zip(dna, source))
        self.assertLessEqual(changed, len(dna) // 7)
        self.assertTrue(
            all((left in "CG") == (right in "CG") for left, right in zip(dna, source))
        )
        recovered = codec.decode(dna, expected_bytes=len(self.payload), sample_id="sample-3")
        self.assertEqual(recovered, self.payload)


@unittest.skipUnless(
    HAS_CRYPTOGRAPHY and HAS_NUMPY,
    "numpy and cryptography are required for the mixed-codec dataset test",
)
class MixedCodecDatasetTest(unittest.TestCase):
    def test_builds_audits_and_decodes_variable_spans(self) -> None:
        import numpy as np
        from PIL import Image

        from genome_mining.image_dataset import (
            audit_image_genome_dataset,
            build_image_genome_dataset,
            read_dataset_metadata,
        )
        from genome_mining.image_payload import RawImageDNASpec, preprocess_image, token_values_to_dna
        from genome_mining.stego_metrics import summarize_stego_dataset
        from genome_mining.stego_validation import verify_stego_roundtrip_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_rows = []
            for split, pattern in (("train", "ACGT"), ("val", "AGCT"), ("test", "ATGC")):
                fasta = root / f"{split}.fa"
                fasta.write_text(">chr1\n" + pattern * 512 + "\n", encoding="utf-8")
                manifest_rows.append(
                    {
                        "donor_id": f"donor_{split}",
                        "assembly_id": split,
                        "fasta": str(fasta),
                        "split": split,
                    }
                )
            genome_manifest = root / "genomes.csv"
            with genome_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            image_root = root / "images"
            image_root.mkdir()
            for index in range(3):
                Image.new("L", (6, 5), color=30 + 50 * index).save(image_root / f"{index}.png")
            key = b"mixed-codec-dataset-key-material"
            dataset = root / "dataset"
            build_image_genome_dataset(
                genome_manifest,
                dataset,
                image_root=image_root,
                n_train=2,
                n_val=2,
                n_test=2,
                positive_fraction=0.5,
                image_size=4,
                window_length=512,
                train_codecs=["direct"],
                val_codecs=["constrained"],
                test_codecs=["kmer"],
                codec_key=key,
                min_acgt_fraction=1.0,
                seed=11,
            )
            self.assertTrue(audit_image_genome_dataset(dataset)["passed"])
            roundtrip = verify_stego_roundtrip_dataset(
                dataset,
                root / "roundtrip",
                codec_key=key,
            )
            self.assertTrue(roundtrip["passed"])
            self.assertEqual(roundtrip["n_checked"], 3)
            summary = summarize_stego_dataset(dataset, root / "summary")
            self.assertEqual({row["codec"] for row in summary["by_codec"]}, {
                "direct_2bit_atcg_v1",
                "constrained_block_v1",
                "cover_kmer_block_v1",
            })
            sequences = np.load(dataset / "sequences.npy", mmap_mode="r")
            for row in read_dataset_metadata(dataset):
                if int(row["y"]) == 0:
                    continue
                start, end = int(row["payload_start"]), int(row["payload_end"])
                dna = token_values_to_dna(sequences[int(row["array_index"]), start:end])
                codec = make_stego_codec(row["codec"], key=key)
                recovered = codec.decode(dna, expected_bytes=16, sample_id=row["sample_id"])
                reference = preprocess_image(row["image_path"], RawImageDNASpec(width=4, height=4)).tobytes()
                self.assertEqual(recovered, reference)
            del sequences


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for the HyenaDNA adapter smoke test")
class HyenaDNAAdapterTest(unittest.TestCase):
    def test_adapter_uses_official_token_ids_and_rc_fusion(self) -> None:
        import types

        import torch
        from torch import nn

        from genome_mining.models.hyenadna_locator import (
            HyenaDNALocator,
            reverse_complement_tokens,
            to_hyenadna_token_ids,
        )

        class DummyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(d_model=16, max_seq_len=1024)
                self.embedding = nn.Embedding(12, 16)

            def forward(self, input_ids, return_dict=True):
                return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        tokens = torch.tensor([[0, 1, 2, 3, 4]])
        self.assertEqual(to_hyenadna_token_ids(tokens).tolist(), [[7, 8, 9, 10, 11]])
        self.assertEqual(reverse_complement_tokens(tokens).tolist(), [[4, 0, 1, 2, 3]])
        model = HyenaDNALocator(
            downsample_stride=16,
            head_dim=16,
            reverse_complement=True,
            fine_tune_mode="frozen",
            backbone_module=DummyBackbone(),
        )
        output = model(torch.randint(0, 5, (2, 128)))
        self.assertEqual(tuple(output["start_bin_logits"].shape), (2, 8))
        self.assertEqual(tuple(output["end_offset_logits"].shape), (2, 8, 16))


if __name__ == "__main__":
    unittest.main()
