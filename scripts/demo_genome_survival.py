from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genome_survival_sim.simulate import SimulationConfig, run_simulation, write_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small genome payload survival demo.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/genome_survival_demo"))
    parser.add_argument("--payload-text", default="hello DNA lineage payload")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--species", nargs="*", default=["toy", "human_t2t", "mouse", "fly"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = args.payload_text.encode("utf-8")

    for species in args.species:
        for strategy in ("single_heterozygous_copy", "multi_locus_redundant"):
            copies_per_block = 1 if strategy == "single_heterozygous_copy" else 16
            result = run_simulation(
                payload,
                SimulationConfig(
                    species=species,
                    generations=args.generations,
                    replicates=args.replicates,
                    strategy=strategy,
                    copies_per_block=copies_per_block,
                    seed=17,
                ),
            )
            out_dir = args.out_dir / f"{species}_{strategy}"
            write_result(result, out_dir)
            print(
                f"{species} {strategy}: "
                f"final_decode_success_rate={result.summary['final_decode_success_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
