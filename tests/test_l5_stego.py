from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

from genome_mining.lm_stego import (
    ContextArithmeticCoder,
    IntegerDistributionConfig,
    KmerProbabilityModel,
)
from genome_mining.lm_stego.integer_distribution import (
    probabilities_to_integer_distribution,
)
from genome_mining.stego_codecs import available_stego_codecs, make_stego_codec


HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _model() -> KmerProbabilityModel:
    return KmerProbabilityModel.fit(
        [
            "ACGTTGCA" * 200,
            "AAAACCCCGGGGTTTT" * 100,
            "AGCTATGCGATCTACG" * 100,
        ],
        order=4,
        minimum_context_count=2,
    )


class IntegerDistributionTest(unittest.TestCase):
    def test_quantization_is_positive_deterministic_and_normalized(self) -> None:
        config = IntegerDistributionConfig(precision_bits=10, uniform_mix=0.2)
        first = probabilities_to_integer_distribution([0.7, 0.2, 0.09, 0.01], config)
        second = probabilities_to_integer_distribution([0.7, 0.2, 0.09, 0.01], config)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.frequencies), 1 << 10)
        self.assertTrue(all(value > 0 for value in first.frequencies))


class KmerProbabilityModelTest(unittest.TestCase):
    def test_saved_model_has_stable_fingerprint_and_probabilities(self) -> None:
        model = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_model.json"
            model.save(path)
            loaded = KmerProbabilityModel.load(path)
        self.assertEqual(loaded.model_id, model.model_id)
        self.assertEqual(loaded.probabilities("ACGTAC"), model.probabilities("ACGTAC"))
        self.assertEqual(loaded.metadata()["n_contexts"], model.metadata()["n_contexts"])


class ContextArithmeticCoderTest(unittest.TestCase):
    def test_exact_round_trip_for_full_and_partial_blocks(self) -> None:
        coder = ContextArithmeticCoder(
            _model(),
            block_bytes=4,
            max_symbols_per_block=192,
            distribution_config=IntegerDistributionConfig(uniform_mix=0.2),
        )
        context = "ACGTTGCAACGT"
        for payload in (b"x", b"hello", bytes(range(31)), b"\x00\xff" * 16):
            with self.subTest(length=len(payload)):
                encoded = coder.encode(payload, context=context)
                recovered = coder.decode(
                    encoded.dna,
                    expected_bytes=len(payload),
                    context=context,
                )
                self.assertEqual(recovered, payload)
                self.assertGreater(len(encoded.dna), 0)
                self.assertGreater(encoded.effective_bits_per_base, 0.0)

    def test_self_delimiting_blocks_decode_without_expected_length(self) -> None:
        coder = ContextArithmeticCoder(
            _model(),
            block_bytes=4,
            max_symbols_per_block=192,
            distribution_config=IntegerDistributionConfig(uniform_mix=0.2),
        )
        payload = bytes(range(32))
        encoded = coder.encode(payload, context="TGCATGCA")
        self.assertEqual(coder.decode(encoded.dna, context="TGCATGCA"), payload)


class L5RegistryTest(unittest.TestCase):
    def test_l5_codec_is_listed_and_requires_a_model(self) -> None:
        rows = {row["name"]: row for row in available_stego_codecs()}
        self.assertIn("lm_arithmetic_v1", rows)
        self.assertEqual(rows["lm_arithmetic_v1"]["level"], "L5-prototype")
        with self.assertRaisesRegex(ValueError, "train-l5-context-model"):
            make_stego_codec("l5", key=b"a sufficiently long test key")


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "cryptography is required for the authenticated L5 codec")
class AuthenticatedL5CodecTest(unittest.TestCase):
    def test_authenticated_l5_packet_round_trip(self) -> None:
        codec = make_stego_codec(
            "l5",
            key=b"authenticated-l5-test-key-material",
            lm_model=_model(),
            lm_block_bytes=4,
            lm_max_symbols_per_block=192,
            lm_uniform_mix=0.2,
        )
        payload = bytes(range(64))
        context = "ACGTTGCA" * 4
        dna = codec.encode(payload, sample_id="sample-l5", context=context)
        recovered = codec.decode(
            dna,
            expected_bytes=len(payload),
            sample_id="sample-l5",
            context=context,
        )
        self.assertEqual(recovered, payload)


@unittest.skipUnless(
    HAS_CRYPTOGRAPHY and HAS_NUMPY,
    "numpy and cryptography are required for the L5 dataset integration test",
)
class L5DatasetIntegrationTest(unittest.TestCase):
    def test_l5_dataset_records_context_model_and_recovers_exact_pixels(self) -> None:
        import numpy as np
        from PIL import Image

        from genome_mining.image_dataset import (
            audit_image_genome_dataset,
            build_image_genome_dataset,
            read_dataset_metadata,
        )
        from genome_mining.image_payload import (
            RawImageDNASpec,
            preprocess_image,
            token_values_to_dna,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome_rows = []
            training_sequences = []
            for split, pattern in (("train", "ACGT"), ("val", "AGCT"), ("test", "ATGC")):
                sequence = pattern * 3000
                training_sequences.append(sequence)
                fasta = root / f"{split}.fa"
                fasta.write_text(f">chr1\n{sequence}\n", encoding="ascii")
                genome_rows.append(
                    {
                        "donor_id": f"donor_{split}",
                        "assembly_id": split,
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
            for index in range(3):
                Image.new("L", (5, 6), color=40 + index * 60).save(image_root / f"{index}.png")

            model_path = root / "context_model.json"
            KmerProbabilityModel.fit(training_sequences, order=3).save(model_path)
            key = b"l5-dataset-integration-key-material"
            dataset = root / "dataset"
            manifest = build_image_genome_dataset(
                genome_manifest,
                dataset,
                image_root=image_root,
                n_train=2,
                n_val=2,
                n_test=2,
                positive_fraction=0.5,
                image_size=4,
                window_length=4096,
                codecs=["lm"],
                codec_key=key,
                lm_model_path=model_path,
                lm_block_bytes=4,
                lm_max_symbols_per_block=128,
                lm_context_bases=32,
                lm_uniform_mix=0.2,
                min_acgt_fraction=1.0,
                seed=17,
            )
            self.assertTrue(manifest["lm_stego"]["enabled"])
            self.assertTrue(audit_image_genome_dataset(dataset)["passed"])

            codec = make_stego_codec(
                "lm",
                key=key,
                lm_model_path=model_path,
                lm_block_bytes=4,
                lm_max_symbols_per_block=128,
                lm_context_bases=32,
                lm_uniform_mix=0.2,
            )
            sequences = np.load(dataset / "sequences.npy", mmap_mode="r")
            for row in read_dataset_metadata(dataset):
                if int(row["y"]) == 0:
                    continue
                index = int(row["array_index"])
                start, end = int(row["payload_start"]), int(row["payload_end"])
                dna = token_values_to_dna(sequences[index, start:end])
                context = token_values_to_dna(sequences[index, max(0, start - 32) : start])
                recovered = codec.decode(
                    dna,
                    expected_bytes=16,
                    sample_id=row["sample_id"],
                    context=context,
                )
                reference = preprocess_image(
                    row["image_path"], RawImageDNASpec(width=4, height=4)
                ).tobytes()
                self.assertEqual(recovered, reference)
            del sequences


if __name__ == "__main__":
    unittest.main()
