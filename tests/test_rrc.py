import itertools
import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from genome_mining.lm_stego.model_adapter import KmerProbabilityModel
from genome_mining.lm_stego.rrc import (
    RRCConfig, encode_bits, decode_bits, encode_packet, decode_packet,
    sample_dna, reverse_value,
)
from genome_mining.rrc_cli import audit_rows, context_records


class RRCChecker(unittest.TestCase):
    def setUp(self):
        self.model = KmerProbabilityModel.fit(["ACGTCAGT" * 100], order=2)
        self.key = bytes(range(32))
        self.nonce = bytes(range(16))

    def test_exhaustive_small_messages(self):
        for length in range(1, 7):
            for message in itertools.product("01", repeat=length):
                bits = "".join(message)
                dna, stats = encode_bits(bits, self.model, key=self.key, nonce=self.nonce)
                self.assertEqual(bits, decode_bits(dna, length, self.model, key=self.key,
                                                   nonce=self.nonce, expected_cdf=stats["cdf_sha256"]))

    def test_varied_lengths_contexts(self):
        randomizer = random.Random(23)
        for length in (7, 32, 128, 256):
            for bits in ("0" * length, "1" * length,
                         format(randomizer.getrandbits(length), f"0{length}b")):
                dna, meta = encode_packet(bits, self.model, key=self.key, nonce=self.nonce,
                                          context="ACGTNACG", output_bases=1024)
                self.assertEqual(len(dna), 1024)
                self.assertEqual(decode_packet(dna, meta, self.model, key=self.key), bits)

    def test_tamper_and_wrong_key(self):
        dna, meta = encode_packet("010101", self.model, key=self.key, nonce=self.nonce)
        with self.assertRaisesRegex(ValueError, "authentication"):
            decode_packet(dna, meta, self.model, key=b"z" * 32)
        with self.assertRaisesRegex(ValueError, "authentication"):
            decode_packet(("C" if dna[0] == "A" else "A") + dna[1:], meta, self.model, key=self.key)
        meta["output_bases"] += 1
        with self.assertRaisesRegex(ValueError, "authentication"):
            decode_packet(dna, meta, self.model, key=self.key)

    def test_trace_detects_probability_change(self):
        dna, stats = encode_bits("010101", self.model, key=self.key, nonce=self.nonce)
        other = KmerProbabilityModel.fit(["A" * 200], order=1)
        with self.assertRaisesRegex(ValueError, "CDF trace"):
            decode_bits(dna, 6, other, key=self.key, nonce=self.nonce,
                        expected_cdf=stats["cdf_sha256"])

    def test_budget_and_inputs(self):
        with self.assertRaises(ValueError):
            encode_bits("1" * 128, self.model, key=self.key, nonce=self.nonce,
                        config=RRCConfig(max_bases=1))
        for bits in ("", "102", "1" * 2049):
            with self.assertRaises(ValueError):
                encode_bits(bits, self.model, key=self.key, nonce=self.nonce)
        with self.assertRaises(ValueError):
            encode_packet("0" * 128, self.model, key=self.key, nonce=self.nonce, output_bases=1)

    def test_nonce_changes_output_and_deterministic_replay(self):
        args = dict(key=self.key, nonce=self.nonce, context="ACG")
        a, _ = encode_bits("0101" * 32, self.model, **args)
        b, _ = encode_bits("0101" * 32, self.model, **args)
        c, _ = encode_bits("0101" * 32, self.model, key=self.key, nonce=b"x" * 16, context="ACG")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_half_ties_and_negative_modulo(self):
        self.assertEqual(reverse_value(Fraction(2), Fraction(3), []), 2)
        # Midpoint 1/4 reverse-rotated by 3/4 modulo 1 is 1/2.
        self.assertEqual(reverse_value(Fraction(0), Fraction(1, 2),
                                        [(Fraction(0), Fraction(1), Fraction(3, 4))]), 0)

    def test_ordinary_sampling_distribution(self):
        uniform = KmerProbabilityModel(order=0)
        dna = sample_dna(2048, uniform, key=self.key, nonce=self.nonce)
        for base in "ACGT":
            self.assertLess(abs(dna.count(base) / len(dna) - 0.25), 0.05)

    def test_streaming_windows_and_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.fa"
            path.write_text(">chr1\nACGTAC\nGTAA\n>chr2\nTTTTCCCC\n")
            self.assertEqual(list(context_records(path, 4)),
                             [("chr1", 0, "ACGT"), ("chr1", 4, "ACGT"),
                              ("chr2", 0, "TTTT"), ("chr2", 4, "CCCC")])
        rows = [{"sample_id": "a", "donor_id": "d", "split": "train", "context": "AC", "natural": "GT"},
                {"sample_id": "b", "donor_id": "d", "split": "test", "context": "AA", "natural": "CC"}]
        with self.assertRaisesRegex(ValueError, "Donor leakage"):
            audit_rows(rows)


if __name__ == "__main__":
    unittest.main()
