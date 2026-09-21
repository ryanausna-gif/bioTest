from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def list_stdpopsim_species() -> list[dict[str, Any]]:
    """Return species metadata from stdpopsim if it is installed."""

    try:
        import stdpopsim
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "stdpopsim is required for list-stdpopsim. Install optional dependencies with: "
            "conda install -c conda-forge stdpopsim"
        ) from exc

    species_rows: list[dict[str, Any]] = []
    for species in stdpopsim.all_species():
        species_rows.append(describe_stdpopsim_species(species))
    return species_rows


def describe_stdpopsim_species(species: Any) -> dict[str, Any]:
    genome = getattr(species, "genome", None)
    chromosomes = getattr(genome, "chromosomes", []) if genome is not None else []
    genetic_maps = getattr(species, "genetic_maps", [])
    demographic_models = getattr(species, "demographic_models", [])
    dfes = getattr(species, "dfes", [])

    return {
        "id": getattr(species, "id", None),
        "name": getattr(species, "name", None),
        "common_name": getattr(species, "common_name", None),
        "num_chromosomes": len(chromosomes),
        "chromosomes": ",".join(getattr(chrom, "id", "") for chrom in chromosomes[:10]),
        "num_genetic_maps": len(genetic_maps),
        "genetic_maps": ",".join(getattr(item, "id", "") for item in genetic_maps[:10]),
        "num_demographic_models": len(demographic_models),
        "demographic_models": ",".join(getattr(item, "id", "") for item in demographic_models[:10]),
        "num_dfes": len(dfes),
    }


def write_stdpopsim_species(rows: list[dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "stdpopsim_species.csv"
    json_path = target / "stdpopsim_species.json"
    report_path = target / "stdpopsim_species_report.md"

    fieldnames = [
        "id",
        "name",
        "common_name",
        "num_chromosomes",
        "chromosomes",
        "num_genetic_maps",
        "genetic_maps",
        "num_demographic_models",
        "demographic_models",
        "num_dfes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    report_path.write_text(make_stdpopsim_species_report(rows), encoding="utf-8")
    return csv_path, json_path, report_path


def make_stdpopsim_species_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# stdpopsim Species Catalog",
        "",
        f"- Species available: `{len(rows)}`",
        "",
        "| id | common_name | chromosomes | genetic_maps | demographic_models |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('id')} | "
            f"{row.get('common_name')} | "
            f"{row.get('num_chromosomes')} | "
            f"{row.get('num_genetic_maps')} | "
            f"{row.get('num_demographic_models')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use this catalog to decide when stdpopsim can replace hand-written species presets.",
            "- The next integration step is to select species/contig/genetic-map IDs and pass them into a stdpopsim simulation backend.",
        ]
    )
    return "\n".join(lines)
