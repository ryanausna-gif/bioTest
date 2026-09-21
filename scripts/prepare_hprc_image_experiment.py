from __future__ import annotations

import argparse
import csv
import random
import shlex
from collections import defaultdict
from pathlib import Path, PurePath, PurePosixPath


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"sample_id", "haplotype", "assembly_name", "assembly"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError("HPRC index is missing columns: " + ", ".join(sorted(missing)))
    return rows


def select_assemblies(
    rows: list[dict[str, str]],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    haplotypes: str,
    seed: int,
) -> list[dict[str, str]]:
    excluded = {"CHM13", "GRCh38"}
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        donor = row["sample_id"].strip()
        haplotype = row["haplotype"].strip()
        uri = row["assembly"].strip()
        if donor in excluded or not donor or not uri or haplotype not in {"1", "2"}:
            continue
        grouped[donor].setdefault(haplotype, row)

    required_haplotypes = {"1"} if haplotypes == "hap1" else {"1", "2"}
    donors = sorted(
        donor for donor, donor_rows in grouped.items() if required_haplotypes <= set(donor_rows)
    )
    random.Random(seed).shuffle(donors)
    total = n_train + n_val + n_test
    if len(donors) < total:
        raise ValueError(f"Only {len(donors)} eligible donors are available; requested {total}.")
    assignments = {}
    offset = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for donor in donors[offset : offset + count]:
            assignments[donor] = split
        offset += count

    selected = []
    wanted = ("1",) if haplotypes == "hap1" else ("1", "2")
    for donor in donors[:total]:
        for haplotype in wanted:
            source = grouped[donor][haplotype]
            uri = source["assembly"].strip()
            selected.append(
                {
                    "donor_id": donor,
                    "assembly_id": source["assembly_name"].strip(),
                    "haplotype": haplotype,
                    "split": assignments[donor],
                    "uri": uri,
                    "filename": PurePosixPath(uri).name,
                    "md5_uri": source.get("assembly_md5", "").strip(),
                }
            )
    return selected


def _unpacked_filename(filename: str) -> str:
    return filename[:-3] if filename.endswith(".gz") else filename


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _public_download_uri(uri: str) -> str:
    if not uri.startswith("s3://"):
        return uri
    bucket_and_key = uri[5:].split("/", 1)
    if len(bucket_and_key) != 2:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = bucket_and_key
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _normalize_fasta_dir(fasta_dir: str | PurePath) -> PurePath:
    raw = str(fasta_dir)
    if raw.startswith("/"):
        return PurePosixPath(raw)
    return Path(raw).resolve()


def write_outputs(
    selected: list[dict[str, str]], out_dir: Path, fasta_dir: str | PurePath
) -> None:
    if not selected:
        raise ValueError("At least one assembly must be selected.")
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = _normalize_fasta_dir(fasta_dir)

    plan_path = out_dir / "hprc_download_plan.tsv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    donor_split_path = out_dir / "donor_split.csv"
    donor_rows: dict[str, dict[str, str | int]] = {}
    for row in selected:
        donor = row["donor_id"]
        if donor not in donor_rows:
            donor_rows[donor] = {
                "donor_id": donor,
                "split": row["split"],
                "assembly_count": 0,
                "haplotypes": "",
            }
        donor_row = donor_rows[donor]
        donor_row["assembly_count"] = int(donor_row["assembly_count"]) + 1
        haplotypes = str(donor_row["haplotypes"]).split(";") if donor_row["haplotypes"] else []
        if row["haplotype"] not in haplotypes:
            haplotypes.append(row["haplotype"])
        donor_row["haplotypes"] = ";".join(haplotypes)
    with donor_split_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["donor_id", "split", "assembly_count", "haplotypes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(donor_rows.values())

    urls_path = out_dir / "hprc_download_urls.txt"
    urls_path.write_text(
        "".join(f'{_public_download_uri(row["uri"])}\n' for row in selected),
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = out_dir / "genome_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["donor_id", "assembly_id", "fasta", "split"]
        )
        writer.writeheader()
        for row in selected:
            filename = _unpacked_filename(row["filename"])
            writer.writerow(
                {
                    "donor_id": row["donor_id"],
                    "assembly_id": row["assembly_id"],
                    "fasta": str(fasta_dir / filename),
                    "split": row["split"],
                }
            )

    linux_download_path = out_dir / "download_and_unpack_hprc.sh"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"FASTA_DIR={shlex.quote(str(fasta_dir))}",
        'mkdir -p "$FASTA_DIR"',
        'if command -v pigz >/dev/null 2>&1; then GZIP_TOOL=pigz; else GZIP_TOOL=gzip; fi',
    ]
    for row in selected:
        uri = shlex.quote(_public_download_uri(row["uri"]))
        target = shlex.quote(str(fasta_dir / row["filename"]))
        download_command = (
            f"curl -L --fail --retry 8 --retry-delay 5 -C - -o {target} {uri}"
        )
        if row["filename"].endswith(".gz"):
            unpacked = shlex.quote(str(fasta_dir / row["filename"][:-3]))
            commands.extend(
                [
                    f"if [[ ! -s {target} ]] || ! \"$GZIP_TOOL\" -t {target} 2>/dev/null; then",
                    f"  {download_command}",
                    "fi",
                    f'"$GZIP_TOOL" -t {target}',
                    f"if [[ ! -s {unpacked} ]]; then",
                    f'  "$GZIP_TOOL" -dk {target}',
                    "fi",
                ]
            )
        else:
            commands.extend(
                [
                    f"if [[ ! -s {target} ]]; then",
                    f"  {download_command}",
                    "fi",
                ]
            )
    commands.extend(
        [
            'echo "Downloaded and unpacked assemblies:"',
            'find "$FASTA_DIR" -maxdepth 1 -type f ! -name "*.gz" -printf "%f\\n" | sort',
        ]
    )
    linux_download_path.write_text(
        "\n".join(commands) + "\n", encoding="utf-8", newline="\n"
    )

    windows_download_path = out_dir / "download_hprc_windows.ps1"
    powershell_rows = []
    for row in selected:
        md5_uri = (
            _public_download_uri(row["md5_uri"])
            if row.get("md5_uri")
            else ""
        )
        powershell_rows.append(
            "    [PSCustomObject]@{ Uri = "
            + _powershell_quote(_public_download_uri(row["uri"]))
            + "; Md5Uri = "
            + _powershell_quote(md5_uri)
            + "; FileName = "
            + _powershell_quote(row["filename"])
            + " }"
        )
    powershell_script = [
        "param(",
        '    [string]$OutputDir = (Join-Path $PSScriptRoot "fasta")',
        ")",
        "",
        '$ErrorActionPreference = "Stop"',
        "$items = @(",
        ",\n".join(powershell_rows),
        ")",
        "",
        "if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {",
        '    throw "curl.exe is required. It is included with current Windows versions."',
        "}",
        "function Get-ExpectedMd5([object]$Item, [string]$Target) {",
        "    if ([string]::IsNullOrWhiteSpace($Item.Md5Uri)) { return $null }",
        '    $md5Path = "$Target.md5"',
        "    & curl.exe -L --fail --retry 5 --retry-delay 5 --output $md5Path $Item.Md5Uri",
        '    if ($LASTEXITCODE -ne 0) { throw "Failed to download MD5 for $($Item.FileName)" }',
        "    $tokens = ((Get-Content -LiteralPath $md5Path -Raw).Trim() -split '\\s+')",
        '    if ($tokens.Count -eq 0 -or $tokens[0] -notmatch "^[0-9a-fA-F]{32}$") {',
        '        throw "Invalid MD5 file for $($Item.FileName): $md5Path"',
        "    }",
        "    return $tokens[0].ToLowerInvariant()",
        "}",
        "function Test-ExpectedMd5([string]$Target, [string]$ExpectedMd5) {",
        "    if (-not (Test-Path -LiteralPath $Target)) { return $false }",
        "    if ((Get-Item -LiteralPath $Target).Length -eq 0) { return $false }",
        "    if ([string]::IsNullOrWhiteSpace($ExpectedMd5)) { return $true }",
        "    $actual = (Get-FileHash -LiteralPath $Target -Algorithm MD5).Hash.ToLowerInvariant()",
        "    return $actual -eq $ExpectedMd5",
        "}",
        "New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null",
        "",
        "foreach ($item in $items) {",
        "    $target = Join-Path $OutputDir $item.FileName",
        "    $expectedMd5 = Get-ExpectedMd5 $item $target",
        "    if (Test-ExpectedMd5 $target $expectedMd5) {",
        '        Write-Host "Already complete and verified: $target"',
        "        continue",
        "    }",
        '    Write-Host "Downloading or resuming: $($item.Uri)"',
        "    & curl.exe -L --fail --retry 8 --retry-delay 5 -C - --output $target $item.Uri",
        "    if ($LASTEXITCODE -ne 0) { throw \"curl.exe failed for $($item.Uri)\" }",
        "    if (-not (Test-ExpectedMd5 $target $expectedMd5)) {",
        '        throw "Downloaded file failed MD5 verification: $target"',
        "    }",
        "}",
        "",
        '$sha256Path = Join-Path $OutputDir "windows_sha256s.txt"',
        "$sha256Lines = foreach ($item in $items) {",
        "    $target = Join-Path $OutputDir $item.FileName",
        "    $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()",
        '    "$hash  $($item.FileName)"',
        "}",
        "$sha256Lines | Set-Content -LiteralPath $sha256Path -Encoding Ascii",
        'Write-Host "SHA256 transfer manifest: $sha256Path"',
        'Write-Host "Complete: $($items.Count) FASTA archives are ready in $OutputDir"',
    ]
    windows_download_path.write_text(
        "\n".join(powershell_script) + "\n", encoding="utf-8-sig", newline="\r\n"
    )

    offline_verify_path = out_dir / "verify_and_unpack_hprc_offline.sh"
    verify_commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"FASTA_DIR={shlex.quote(str(fasta_dir))}",
        f"EXPECTED_COUNT={len(selected)}",
        'mkdir -p "$FASTA_DIR"',
        'if command -v pigz >/dev/null 2>&1; then GZIP_TOOL=pigz; else GZIP_TOOL=gzip; fi',
        "missing=0",
    ]
    for row in selected:
        archive = shlex.quote(str(fasta_dir / row["filename"]))
        verify_commands.extend(
            [
                f"if [[ ! -s {archive} ]]; then",
                f"  echo \"Missing archive: {archive}\" >&2",
                "  missing=1",
                "fi",
            ]
        )
    verify_commands.extend(
        [
            'if [[ "$missing" -ne 0 ]]; then exit 2; fi',
            'echo "All expected archives are present. Validating and unpacking..."',
        ]
    )
    for row in selected:
        if not row["filename"].endswith(".gz"):
            continue
        archive = shlex.quote(str(fasta_dir / row["filename"]))
        unpacked = shlex.quote(str(fasta_dir / _unpacked_filename(row["filename"])))
        verify_commands.extend(
            [
                f'"$GZIP_TOOL" -t {archive}',
                f"if [[ ! -s {unpacked} ]]; then",
                f'  "$GZIP_TOOL" -dk {archive}',
                "fi",
                f"[[ -s {unpacked} ]] || {{ echo \"Unpacked FASTA is missing: {unpacked}\" >&2; exit 3; }}",
            ]
        )
    verify_commands.extend(
        [
            'actual=$(find "$FASTA_DIR" -maxdepth 1 -type f ! -name "*.gz" | wc -l)',
            'echo "Expected assemblies: $EXPECTED_COUNT"',
            'echo "Available unpacked files: $actual"',
            'echo "Offline HPRC import completed successfully."',
        ]
    )
    offline_verify_path.write_text(
        "\n".join(verify_commands) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"selected_assemblies={len(selected)}")
    print(f"download_plan={plan_path}")
    print(f"donor_split={donor_split_path}")
    print(f"download_urls={urls_path}")
    print(f"linux_download_script={linux_download_path}")
    print(f"windows_download_script={windows_download_path}")
    print(f"offline_verify_script={offline_verify_path}")
    print(f"genome_manifest={manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select donor-disjoint HPRC Release 2 assemblies for the image localization experiment."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fasta-dir",
        required=True,
        help="Final FASTA directory. POSIX absolute paths are preserved when run on Windows.",
    )
    parser.add_argument("--n-train-donors", type=int, default=8)
    parser.add_argument("--n-val-donors", type=int, default=2)
    parser.add_argument("--n-test-donors", type=int, default=2)
    parser.add_argument("--haplotypes", choices=["hap1", "both"], default="hap1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = (args.n_train_donors, args.n_val_donors, args.n_test_donors)
    if any(count <= 0 for count in counts):
        raise ValueError("Each split must contain at least one donor.")
    rows = _read_index(args.index)
    selected = select_assemblies(
        rows,
        n_train=args.n_train_donors,
        n_val=args.n_val_donors,
        n_test=args.n_test_donors,
        haplotypes=args.haplotypes,
        seed=args.seed,
    )
    write_outputs(selected, args.out, args.fasta_dir)


if __name__ == "__main__":
    main()
