from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from typing import Iterable


RUN_KEYS = (
    "species",
    "strategy",
    "copies_per_block",
    "block_size_bytes",
    "encryption_mode",
    "ecc_mode",
    "ecc_symbols",
    "erasure_mode",
    "erasure_group_size",
    "erasure_repair_blocks",
    "resync_mode",
    "resync_chunk_bytes",
    "resync_marker_bp",
    "resync_mismatches",
    "mutation_rate_multiplier",
)


def build_batch_report(summaries: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    half_life = estimate_half_life(rows)
    tradeoffs = build_design_tradeoffs(summaries, rows)
    pareto = [record for record in tradeoffs if record["pareto_efficient"]]
    ranked = sorted(
        summaries,
        key=lambda item: float(item.get("final_decode_success_rate", 0.0)),
        reverse=True,
    )

    lines = [
        "# 批量仿真总报告",
        "",
        "## 最优组合",
        "",
        "| rank | species | strategy | encryption_mode | ecc_mode | ecc_symbols | erasure_mode | erasure_group_size | erasure_repair_blocks | resync_mode | resync_chunk_bytes | resync_marker_bp | resync_mismatches | copies_per_block | mutx | generations | final_decode_success_rate | half_life_generation | top_loss_reason |",
        "|---:|---|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, summary in enumerate(ranked[:10], start=1):
        key = _summary_key(summary)
        lines.append(
            "| "
            f"{index} | "
            f"{summary.get('species')} | "
            f"{summary.get('strategy')} | "
            f"{summary.get('encryption_mode')} | "
            f"{summary.get('ecc_mode')} | "
            f"{summary.get('ecc_symbols')} | "
            f"{summary.get('erasure_mode')} | "
            f"{summary.get('erasure_group_size')} | "
            f"{summary.get('erasure_repair_blocks')} | "
            f"{summary.get('resync_mode')} | "
            f"{summary.get('resync_chunk_bytes')} | "
            f"{summary.get('resync_marker_bp')} | "
            f"{summary.get('resync_mismatches')} | "
            f"{summary.get('copies_per_block')} | "
            f"{summary.get('mutation_rate_multiplier')} | "
            f"{summary.get('generations')} | "
            f"{float(summary.get('final_decode_success_rate', 0.0)):.4f} | "
            f"{half_life.get(key, 'NA')} | "
            f"{summary.get('top_final_loss_reason')} |"
        )

    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- `half_life_generation` 表示完整解码率首次降到 0.5 或以下的世代；`NA` 表示在本次世代范围内未降到 0.5。",
            "- 如果最佳组合的 `top_loss_reason` 仍然是 `not_inherited`，说明优先级仍是遗传稳定性和拷贝布局，而不是继续堆突变纠错。",
            "- 如果 `auth_failed` 或 `block_missing_or_mutated` 上升，才说明需要优先增强纠错码、标识符冗余和 block 级擦除恢复。",
            "",
        ]
    )
    if tradeoffs:
        lines.extend(
            [
                "## 成本-收益前沿",
                "",
                "生成文件：`batch_design_tradeoffs.csv` 和 `batch_tradeoff.svg`。",
                "",
                "| rank | pareto | species | strategy | encoded_total_bp | bp_per_payload_byte | final_decode_success_rate | half_life_generation | config |",
                "|---:|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for index, record in enumerate(pareto[:10], start=1):
            config = (
                f"copies={record['copies_per_block']}, "
                f"mutx={record['mutation_rate_multiplier']}, "
                f"ecc={record['ecc_mode']}/{record['ecc_symbols']}, "
                f"erasure={record['erasure_mode']}/{record['erasure_group_size']}/{record['erasure_repair_blocks']}, "
                f"resync={record['resync_mode']}/{record['resync_chunk_bytes']}"
            )
            lines.append(
                "| "
                f"{index} | "
                f"{'yes' if record['pareto_efficient'] else 'no'} | "
                f"{record['species']} | "
                f"{record['strategy']} | "
                f"{int(float(record['encoded_total_bp']))} | "
                f"{float(record['encoded_bp_per_payload_byte']):.2f} | "
                f"{float(record['final_decode_success_rate']):.4f} | "
                f"{record['half_life_generation']} | "
                f"{config} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_design_tradeoffs(summaries: list[dict[str, object]], rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    half_life = estimate_half_life(rows)
    generation_zero = _generation_zero_by_key(rows)
    records: list[dict[str, object]] = []

    for summary in summaries:
        key = _summary_key(summary)
        zero = generation_zero.get(key, {})
        encoded_total_bp = _float(summary.get("encoded_total_bp", zero.get("mean_copy_count", 0.0)))
        payload_bytes = _float(summary.get("payload_bytes", 0.0))
        record = {
            "species": summary.get("species"),
            "strategy": summary.get("strategy"),
            "copies_per_block": summary.get("copies_per_block"),
            "block_size_bytes": summary.get("block_size_bytes"),
            "encryption_mode": summary.get("encryption_mode"),
            "ecc_mode": summary.get("ecc_mode"),
            "ecc_symbols": summary.get("ecc_symbols"),
            "erasure_mode": summary.get("erasure_mode"),
            "erasure_group_size": summary.get("erasure_group_size"),
            "erasure_repair_blocks": summary.get("erasure_repair_blocks"),
            "resync_mode": summary.get("resync_mode"),
            "resync_chunk_bytes": summary.get("resync_chunk_bytes"),
            "resync_marker_bp": summary.get("resync_marker_bp"),
            "resync_mismatches": summary.get("resync_mismatches"),
            "mutation_rate_multiplier": summary.get("mutation_rate_multiplier"),
            "generations": summary.get("generations"),
            "replicates": summary.get("replicates"),
            "payload_bytes": payload_bytes,
            "encoded_fragment_count": _float(summary.get("encoded_fragment_count", zero.get("mean_copy_count", 0.0))),
            "encoded_total_bp": encoded_total_bp,
            "encoded_bp_per_payload_byte": _float(
                summary.get("encoded_bp_per_payload_byte", encoded_total_bp / max(1.0, payload_bytes))
            ),
            "final_decode_success_rate": _float(summary.get("final_decode_success_rate", 0.0)),
            "final_has_any_copy_rate": _float(summary.get("final_has_any_copy_rate", 0.0)),
            "final_identifier_detected_rate": _float(summary.get("final_identifier_detected_rate", 0.0)),
            "half_life_generation": half_life.get(key, "NA"),
            "top_final_loss_reason": summary.get("top_final_loss_reason"),
            "pareto_efficient": False,
        }
        records.append(record)

    _mark_pareto(records)
    records.sort(
        key=lambda item: (
            not bool(item["pareto_efficient"]),
            -float(item["final_decode_success_rate"]),
            float(item["encoded_total_bp"]),
        )
    )
    return records


def write_design_tradeoff_artifacts(
    summaries: list[dict[str, object]],
    rows: Iterable[dict[str, object]],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    records = build_design_tradeoffs(summaries, rows)

    csv_path = target / "batch_design_tradeoffs.csv"
    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    svg_path = target / "batch_tradeoff.svg"
    svg_path.write_text(build_tradeoff_svg(records), encoding="utf-8")
    return csv_path, svg_path


def build_tradeoff_svg(records: list[dict[str, object]], width: int = 900, height: int = 520) -> str:
    if not records:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>\n'

    left, right, top, bottom = 78, 28, 28, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    costs = [max(0.0, _float(record.get("encoded_total_bp"))) for record in records]
    rates = [min(1.0, max(0.0, _float(record.get("final_decode_success_rate")))) for record in records]
    min_cost, max_cost = min(costs), max(costs)
    if min_cost == max_cost:
        min_cost = max(0.0, min_cost - 1.0)
        max_cost += 1.0

    def x_for(cost: float) -> float:
        return left + ((cost - min_cost) / (max_cost - min_cost)) * plot_w

    def y_for(rate: float) -> float:
        return top + (1.0 - rate) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">encoded_total_bp</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">final_decode_success_rate</text>',
        f'<text x="{left}" y="20" font-family="Arial" font-size="15" font-weight="bold">Payload survival cost-benefit</text>',
    ]

    for tick in range(6):
        rate = tick / 5
        y = y_for(rate)
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{rate:.1f}</text>'
        )
    for tick in range(5):
        frac = tick / 4
        cost = min_cost + frac * (max_cost - min_cost)
        x = x_for(cost)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#222"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{cost:.0f}</text>'
        )

    pareto_points = sorted(
        [record for record in records if record.get("pareto_efficient")],
        key=lambda item: _float(item.get("encoded_total_bp")),
    )
    if len(pareto_points) >= 2:
        points = " ".join(
            f"{x_for(_float(item.get('encoded_total_bp'))):.1f},{y_for(_float(item.get('final_decode_success_rate'))):.1f}"
            for item in pareto_points
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="#111" stroke-width="1.8" stroke-dasharray="4 3"/>')

    palette = {
        "single_heterozygous_copy": "#D55E00",
        "homozygous_same_locus": "#0072B2",
        "multi_locus_redundant": "#009E73",
    }
    for record in records:
        cost = _float(record.get("encoded_total_bp"))
        rate = _float(record.get("final_decode_success_rate"))
        strategy = str(record.get("strategy"))
        color = palette.get(strategy, "#6A4C93")
        radius = 6 if record.get("pareto_efficient") else 4
        stroke = "#111" if record.get("pareto_efficient") else "#ffffff"
        title = html.escape(
            f"{record.get('species')} {strategy} success={rate:.4f} bp={cost:.0f} "
            f"copies={record.get('copies_per_block')} erasure={record.get('erasure_mode')}"
        )
        parts.append(
            f'<circle cx="{x_for(cost):.1f}" cy="{y_for(rate):.1f}" r="{radius}" fill="{color}" '
            f'stroke="{stroke}" stroke-width="1.4"><title>{title}</title></circle>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def estimate_half_life(rows: Iterable[dict[str, object]], threshold: float = 0.5) -> dict[tuple[object, ...], int]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_row_key(row)].append(row)

    result: dict[tuple[object, ...], int] = {}
    for key, items in grouped.items():
        items.sort(key=lambda row: int(row["generation"]))
        for row in items:
            if float(row["decode_success_rate"]) <= threshold:
                result[key] = int(row["generation"])
                break
    return result


def _generation_zero_by_key(rows: Iterable[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if int(row.get("generation", -1)) == 0:
            result[_row_key(row)] = row
    return result


def _mark_pareto(records: list[dict[str, object]]) -> None:
    for index, record in enumerate(records):
        cost = _float(record.get("encoded_total_bp"))
        success = _float(record.get("final_decode_success_rate"))
        dominated = False
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            other_cost = _float(other.get("encoded_total_bp"))
            other_success = _float(other.get("final_decode_success_rate"))
            if other_cost <= cost and other_success >= success and (other_cost < cost or other_success > success):
                dominated = True
                break
        record["pareto_efficient"] = not dominated


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row.get(key) for key in RUN_KEYS)


def _summary_key(summary: dict[str, object]) -> tuple[object, ...]:
    return tuple(summary.get(key) for key in RUN_KEYS)
