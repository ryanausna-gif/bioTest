from __future__ import annotations

import unittest
from random import Random

from image_format_recovery.formats import recover_image
from image_format_recovery.synthetic import available_formats, make_sample


class BaselineRecoveryTest(unittest.TestCase):
    def test_recovers_common_formats_with_noise_prefix(self) -> None:
        rng = Random(42)
        formats = available_formats(["png", "jpeg", "bmp", "webp"])
        self.assertGreaterEqual(len(formats), 3)

        for format_name in formats:
            with self.subTest(format_name=format_name):
                sample = make_sample(rng, format_name=format_name, max_prefix=64, max_suffix=32)
                image, candidate = recover_image(sample.stream)
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.format_name, format_name)
                self.assertEqual(candidate.start, sample.start_offset)
                self.assertEqual(image.size, (sample.width, sample.height))


if __name__ == "__main__":
    unittest.main()
