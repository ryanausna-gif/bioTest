from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.prepare_oxford_pet_manifest import prepare_manifest


class PrepareOxfordPetManifestTest(unittest.TestCase):
    def test_builds_deterministic_disjoint_stratified_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            annotation_lines = ["# image class species breed"]
            for class_id in range(1, 4):
                for index in range(1, 7):
                    image_id = f"breed_{class_id}_{index}"
                    (images / f"{image_id}.jpg").write_bytes(
                        f"image-{class_id}-{index}".encode("ascii")
                    )
                    annotation_lines.append(f"{image_id} {class_id} 1 {class_id}")
            annotations = root / "list.txt"
            annotations.write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")

            output = root / "image_manifest.csv"
            report = prepare_manifest(
                images,
                annotations,
                output,
                n_train=6,
                n_val=3,
                n_test=6,
                seed=2026,
            )
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(report["split_counts"], {"train": 6, "val": 3, "test": 6})
            self.assertEqual(len(rows), 15)
            self.assertEqual(len({row["image_id"] for row in rows}), 15)
            self.assertEqual(len({row["source_sha256"] for row in rows}), 15)
            self.assertTrue(all(Path(row["path"]).is_absolute() for row in rows))
            for split, expected in (("train", 6), ("val", 3), ("test", 6)):
                counts = Counter(row["class_id"] for row in rows if row["split"] == split)
                self.assertEqual(sum(counts.values()), expected)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

            repeated = root / "repeated.csv"
            repeated_report = prepare_manifest(
                images,
                annotations,
                repeated,
                n_train=6,
                n_val=3,
                n_test=6,
                seed=2026,
            )
            self.assertEqual(output.read_text(encoding="utf-8"), repeated.read_text(encoding="utf-8"))
            self.assertEqual(report["class_counts"], repeated_report["class_counts"])


if __name__ == "__main__":
    unittest.main()
