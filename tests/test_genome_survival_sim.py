from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from genome_survival_sim.codec import (
    PayloadCodec,
    PayloadCodecConfig,
    normalize_ecc_mode,
    normalize_encryption_mode,
    normalize_erasure_mode,
    normalize_resync_mode,
)
from genome_survival_sim.analysis import build_design_tradeoffs, build_tradeoff_svg, estimate_half_life
from genome_survival_sim.batch import run_batch
from genome_survival_sim.genome_io import build_candidate_intervals, load_chrom_sizes_dict, load_gff_features
from genome_survival_sim.msprime_backend import (
    MsprimeConfig,
    expected_payload_mutations,
    make_msprime_report,
    summarize_msprime_tree_sequence,
)
from genome_survival_sim.simulate import SimulationConfig, run_simulation
from genome_survival_sim.slim_backend import (
    SlimBridgeConfig,
    parse_slim_stdout,
    parse_slim_stdout_file,
    render_slim_script,
)
from genome_survival_sim.stdpopsim_backend import make_stdpopsim_species_report
from genome_survival_sim.tskit_backend import extract_slim_mutation_types, summarize_tree_sequence


class FakeMutation:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeTreeSequence:
    sequence_length = 1000
    num_trees = 3
    num_nodes = 10
    num_edges = 9
    num_sites = 2
    num_mutations = 3
    num_individuals = 4
    num_populations = 1
    num_samples = 8

    def mutations(self):
        return iter(
            [
                FakeMutation({"mutation_list": [{"mutation_type": 1}, {"mutation_type": 2}]}),
                FakeMutation({"mutation_type": 1}),
                FakeMutation({}),
            ]
        )


class PayloadCodecTest(unittest.TestCase):
    def test_roundtrip_with_redundant_copy(self) -> None:
        payload = b"hello lineage simulator"
        codec = PayloadCodec(PayloadCodecConfig(block_size_bytes=8, copies_per_block=2))
        expected = codec.with_expected_payload_marker(payload)
        fragments = codec.encode(payload)

        damaged = list(fragments)
        damaged[0] = type(damaged[0])(
            block_id=damaged[0].block_id,
            total_blocks=damaged[0].total_blocks,
            copy_id=damaged[0].copy_id,
            dna="A" + damaged[0].dna[1:],
        )

        result = expected.decode_fragments(fragment.dna for fragment in damaged)
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_normalizes_encryption_modes(self) -> None:
        self.assertEqual(normalize_encryption_mode("simulation"), "hmac_stream")
        self.assertEqual(normalize_encryption_mode("aead"), "chacha20_poly1305")
        with self.assertRaises(ValueError):
            normalize_encryption_mode("unknown")

    def test_chacha20_poly1305_mode_is_optional(self) -> None:
        payload = b"hello aead"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=32,
                copies_per_block=1,
                encryption_mode="chacha20_poly1305",
            )
        )
        if importlib.util.find_spec("cryptography") is None:
            with self.assertRaises(ImportError):
                codec.encode(payload)
            return

        fragments = codec.encode(payload)
        result = codec.decode_fragments(fragment.dna for fragment in fragments)
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_normalizes_ecc_modes(self) -> None:
        self.assertEqual(normalize_ecc_mode("off"), "none")
        self.assertEqual(normalize_ecc_mode("rs"), "reed_solomon")
        with self.assertRaises(ValueError):
            normalize_ecc_mode("unknown")

    def test_reed_solomon_mode_is_optional_and_corrects_small_damage(self) -> None:
        payload = b"hello reed solomon"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=64,
                copies_per_block=1,
                ecc_mode="reed_solomon",
                ecc_symbols=8,
            )
        )
        if importlib.util.find_spec("reedsolo") is None:
            with self.assertRaises(ImportError):
                codec.encode(payload)
            return

        fragment = codec.encode(payload)[0]
        dna = fragment.dna
        damage_index = codec.config.sync_marker_bp + 8
        replacement = "A" if dna[damage_index] != "A" else "C"
        damaged_dna = dna[:damage_index] + replacement + dna[damage_index + 1 :]
        result = codec.decode_fragments([damaged_dna])
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_normalizes_erasure_modes(self) -> None:
        self.assertEqual(normalize_erasure_mode("off"), "none")
        self.assertEqual(normalize_erasure_mode("xor"), "xor_parity")
        self.assertEqual(normalize_erasure_mode("rs_erasure"), "fountain")
        with self.assertRaises(ValueError):
            normalize_erasure_mode("unknown")

    def test_xor_parity_recovers_one_missing_data_block(self) -> None:
        payload = b"0123456789abcdefXYZ"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=8,
                copies_per_block=1,
                erasure_mode="xor_parity",
                erasure_group_size=4,
            )
        )
        fragments = codec.encode(payload)
        kept = [fragment for fragment in fragments if fragment.block_id != 1]

        result = codec.decode_fragments(fragment.dna for fragment in kept)
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_fountain_recovers_two_missing_data_blocks(self) -> None:
        payload = b"block-00block-01block-02block-03"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=8,
                copies_per_block=1,
                erasure_mode="fountain",
                erasure_group_size=4,
                erasure_repair_blocks=4,
            )
        )
        fragments = codec.encode(payload)
        kept = [fragment for fragment in fragments if fragment.block_id not in {1, 2}]

        result = codec.decode_fragments(fragment.dna for fragment in kept)
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_xor_parity_recovers_missing_final_short_block(self) -> None:
        payload = b"short-final-block"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=8,
                copies_per_block=1,
                erasure_mode="xor_parity",
                erasure_group_size=4,
            )
        )
        fragments = codec.encode(payload)
        data_block_ids = [fragment.block_id for fragment in fragments if fragment.block_id < fragment.total_blocks]
        kept = [fragment for fragment in fragments if fragment.block_id != max(data_block_ids)]

        result = codec.decode_fragments(fragment.dna for fragment in kept)
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)

    def test_normalizes_resync_modes(self) -> None:
        self.assertEqual(normalize_resync_mode("off"), "none")
        self.assertEqual(normalize_resync_mode("chunk_sync"), "chunked")
        with self.assertRaises(ValueError):
            normalize_resync_mode("unknown")

    def test_chunked_resync_marks_damaged_chunk_as_erasure(self) -> None:
        packet = b"abcdefghijklmnopqrstuvwxyz"
        codec = PayloadCodec(
            PayloadCodecConfig(
                resync_mode="chunked",
                resync_chunk_bytes=8,
                resync_marker_bp=20,
            )
        )
        body = codec._serialize_packet(packet)
        damaged_body = body[: len(codec._chunk_marker()) + 4] + "A" + body[len(codec._chunk_marker()) + 4 :]

        raw, erase_positions = codec._deserialize_body(damaged_body)
        self.assertEqual(raw[8:], packet[8:])
        self.assertEqual(erase_positions, list(range(8)))

    def test_chunked_resync_with_reed_solomon_can_recover_one_indel_chunk(self) -> None:
        payload = b"chunked indel recovery payload"
        codec = PayloadCodec(
            PayloadCodecConfig(
                block_size_bytes=64,
                copies_per_block=1,
                ecc_mode="reed_solomon",
                ecc_symbols=16,
                resync_mode="chunked",
                resync_chunk_bytes=8,
                resync_marker_bp=20,
            )
        )
        if importlib.util.find_spec("reedsolo") is None:
            with self.assertRaises(ImportError):
                codec.encode(payload)
            return

        fragment = codec.encode(payload)[0]
        marker_len = codec.config.sync_marker_bp
        chunk_marker_len = codec.config.resync_marker_bp
        damage_index = marker_len + chunk_marker_len + 12
        damaged_dna = fragment.dna[:damage_index] + "A" + fragment.dna[damage_index:]

        result = codec.decode_fragments([damaged_dna])
        self.assertTrue(result.success)
        self.assertEqual(result.payload, payload)


class GenomeSurvivalSimulationTest(unittest.TestCase):
    def test_homozygous_founder_survives_first_generation_without_mutation(self) -> None:
        result = run_simulation(
            b"payload",
            SimulationConfig(
                species="toy",
                generations=1,
                replicates=20,
                strategy="homozygous_same_locus",
                mutation_rate_multiplier=0,
                snv_rate=0,
                indel_rate=0,
                seed=3,
                block_size_bytes=16,
            ),
        )
        final = result.rows[-1]
        self.assertEqual(final["decode_success_rate"], 1.0)

    def test_single_heterozygous_copy_loses_some_lineages(self) -> None:
        result = run_simulation(
            b"payload",
            SimulationConfig(
                species="toy",
                generations=4,
                replicates=200,
                strategy="single_heterozygous_copy",
                mutation_rate_multiplier=0,
                snv_rate=0,
                indel_rate=0,
                seed=5,
                block_size_bytes=16,
            ),
        )
        final = result.rows[-1]
        self.assertLess(final["decode_success_rate"], 1.0)
        self.assertGreaterEqual(final["not_inherited"], 1)


class GenomeIoTest(unittest.TestCase):
    def test_builds_intergenic_and_intronic_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sizes = root / "test.chrom.sizes"
            gff = root / "test.gff3"
            sizes.write_text("chr1\t1000\n", encoding="utf-8")
            gff.write_text(
                "\n".join(
                    [
                        "chr1\ttest\tgene\t101\t500\t.\t+\t.\tID=gene1",
                        "chr1\ttest\texon\t101\t200\t.\t+\t.\tParent=gene1",
                        "chr1\ttest\texon\t401\t500\t.\t+\t.\tParent=gene1",
                        "chr1\ttest\tpseudogene\t701\t800\t.\t+\t.\tID=pg1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            chrom_sizes = load_chrom_sizes_dict(sizes)
            features = load_gff_features(gff)
            intergenic = build_candidate_intervals(chrom_sizes, features, mode="intergenic", min_length=50, flank=0)
            intronic = build_candidate_intervals(chrom_sizes, features, mode="intronic", min_length=50, flank=0)
            pseudogene = build_candidate_intervals(chrom_sizes, features, mode="pseudogene", min_length=50, flank=0)

            self.assertTrue(any(item.start == 0 and item.end == 100 for item in intergenic))
            self.assertTrue(any(item.start == 200 and item.end == 400 for item in intronic))
            self.assertEqual(len(pseudogene), 1)


class BatchAndSlimTest(unittest.TestCase):
    def test_estimates_half_life(self) -> None:
        rows = [
            {
                "species": "toy",
                "strategy": "s",
                "copies_per_block": "default",
                "block_size_bytes": 16,
                "mutation_rate_multiplier": 1.0,
                "generation": 0,
                "decode_success_rate": 1.0,
            },
            {
                "species": "toy",
                "strategy": "s",
                "copies_per_block": "default",
                "block_size_bytes": 16,
                "mutation_rate_multiplier": 1.0,
                "generation": 2,
                "decode_success_rate": 0.4,
            },
        ]
        self.assertEqual(next(iter(estimate_half_life(rows).values())), 2)

    def test_builds_tradeoff_records_and_pareto_front(self) -> None:
        summaries = [
            {
                "species": "toy",
                "strategy": "cheap",
                "copies_per_block": 1,
                "block_size_bytes": 16,
                "encoded_total_bp": 100,
                "payload_bytes": 10,
                "final_decode_success_rate": 0.8,
                "mutation_rate_multiplier": 1.0,
            },
            {
                "species": "toy",
                "strategy": "expensive",
                "copies_per_block": 4,
                "block_size_bytes": 16,
                "encoded_total_bp": 200,
                "payload_bytes": 10,
                "final_decode_success_rate": 0.7,
                "mutation_rate_multiplier": 1.0,
            },
        ]
        records = build_design_tradeoffs(summaries, [])
        by_strategy = {record["strategy"]: record for record in records}
        self.assertTrue(by_strategy["cheap"]["pareto_efficient"])
        self.assertFalse(by_strategy["expensive"]["pareto_efficient"])
        self.assertIn("<svg", build_tradeoff_svg(records))

    def test_batch_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "batch"
            run_batch(
                {
                    "payload_text": "abc",
                    "species": "toy",
                    "generations": 1,
                    "replicates": 4,
                    "strategies": ["single_heterozygous_copy"],
                    "mutation_rate_multipliers": [1],
                    "block_size_bytes": 16,
                },
                out,
            )
            summary = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 1)
            self.assertTrue((out / "batch_generation_metrics.csv").exists())
            self.assertTrue((out / "batch_report.md").exists())
            self.assertTrue((out / "batch_design_tradeoffs.csv").exists())
            self.assertTrue((out / "batch_tradeoff.svg").exists())

    def test_renders_slim_script(self) -> None:
        script = render_slim_script(SlimBridgeConfig(genome_length=1000, population_size=20, generations=3))
        self.assertIn("initializeSLiMModelType", script)
        self.assertIn("generation,carrier_genomes", script)

    def test_renders_tree_sequence_output(self) -> None:
        script = render_slim_script(
            SlimBridgeConfig(
                genome_length=1000,
                population_size=20,
                generations=3,
                tree_seq_output="payload_bridge.trees",
            )
        )
        self.assertIn("initializeTreeSeq", script)
        self.assertIn('sim.treeSeqOutput("payload_bridge.trees")', script)

    def test_parses_slim_stdout(self) -> None:
        stdout = "\n".join(
            [
                "// SLiM banner",
                "generation,carrier_genomes,total_payload_markers,marker_frequency",
                "1,12,4,0.2",
                "2,8,3,0.13",
                "noise line",
                "3,0,0,0",
            ]
        )
        rows = parse_slim_stdout(stdout, population_size=10)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["generation"], 3)
        self.assertEqual(rows[-1]["carrier_genome_rate"], 0.0)

    def test_writes_parsed_slim_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout_path = root / "slim_stdout.txt"
            stdout_path.write_text(
                "generation,carrier_genomes,total_payload_markers,marker_frequency\n"
                "1,4,2,0.2\n"
                "2,2,1,0.1\n",
                encoding="utf-8",
            )
            csv_path, summary_path, report_path = parse_slim_stdout_file(
                stdout_path,
                root / "parsed",
                population_size=4,
            )
            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())


class TreeSequenceBackendTest(unittest.TestCase):
    def test_extracts_common_slim_metadata_layouts(self) -> None:
        self.assertEqual(extract_slim_mutation_types({"mutation_type": 3}), [3])
        self.assertEqual(
            extract_slim_mutation_types({"mutation_list": [{"mutation_type": 1}, {"mutation_type_id": 2}]}),
            [1, 2],
        )

    def test_summarizes_tree_sequence_like_object(self) -> None:
        summary = summarize_tree_sequence(FakeTreeSequence(), mutation_type=1)
        self.assertEqual(summary["num_mutations"], 3)
        self.assertEqual(summary["mutation_type_counts"]["1"], 2)
        self.assertEqual(summary["mutation_type_counts"]["2"], 1)
        self.assertEqual(summary["mutation_type_counts"]["unknown"], 1)
        self.assertEqual(summary["filtered_mutations"], 2)


class MsprimeBackendTest(unittest.TestCase):
    def test_expected_payload_mutations(self) -> None:
        self.assertAlmostEqual(expected_payload_mutations(1000, 10, 1e-8), 1e-4)

    def test_summarizes_msprime_like_tree_sequence(self) -> None:
        config = MsprimeConfig(samples=4, sequence_length=1000, mutation_rate=1e-8, payload_bp=200)
        summary = summarize_msprime_tree_sequence(FakeTreeSequence(), config)
        self.assertEqual(summary["backend"], "msprime")
        self.assertEqual(summary["config"]["samples"], 4)
        self.assertEqual(summary["expected_payload_mutations_per_lineage"], 2e-6)
        report = make_msprime_report(summary)
        self.assertIn("msprime Neutral Baseline Report", report)


class StdpopsimBackendTest(unittest.TestCase):
    def test_makes_catalog_report(self) -> None:
        report = make_stdpopsim_species_report(
            [
                {
                    "id": "HomSap",
                    "common_name": "Human",
                    "num_chromosomes": 25,
                    "num_genetic_maps": 2,
                    "num_demographic_models": 4,
                }
            ]
        )
        self.assertIn("stdpopsim Species Catalog", report)
        self.assertIn("HomSap", report)


if __name__ == "__main__":
    unittest.main()
