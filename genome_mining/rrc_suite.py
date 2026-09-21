"""Sequential multi-rate experiment: python -m genome_mining.rrc_suite --config FILE."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .rrc_cli import write_csv
from .rrc_experiment import atomic_json


def resolve_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {"contexts", "key_file", "model", "hf_model", "out", "device", "trust_remote_code",
               "bits", "length", "context_bases", "precision_bits", "max_bases", "detectors",
               "detector_hf_model", "epochs", "batch_size", "seed", "bootstrap", "fpr"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config fields: {sorted(set(config)-allowed)}")
    for field in ("contexts", "key_file", "model", "hf_model", "out", "detector_hf_model"):
        if config.get(field):
            config[field] = str((path.parent / config[field]).resolve())
    if not all(config.get(k) for k in ("contexts", "key_file", "out")):
        raise ValueError("contexts, key_file and out are required")
    if bool(config.get("model")) == bool(config.get("hf_model")):
        raise ValueError("Select exactly one generator: model or hf_model")
    defaults = {"device": "cpu", "bits": [32, 64, 128], "length": 512, "context_bases": 1024,
                "precision_bits": 12, "max_bases": 4096, "detectors": ["logreg", "cnn"],
                "epochs": 10, "batch_size": 32, "seed": 2026, "bootstrap": 500, "fpr": 0.01}
    config = {**defaults, **config}
    if not config["bits"] or any(type(b) is not int or not 1 <= b <= 2048 for b in config["bits"]):
        raise ValueError("bits must be a nonempty list of integers in 1..2048")
    if len(set(config["bits"])) != len(config["bits"]):
        raise ValueError("Duplicate bit rates")
    if not config["detectors"] or set(config["detectors"]) - {"logreg", "cnn", "hyena-probe"}:
        raise ValueError("Invalid detectors")
    if "hyena-probe" in config["detectors"] and not config.get("detector_hf_model"):
        raise ValueError("hyena-probe requires detector_hf_model")
    return config


def run_suite(config, resume=False):
    root = Path(config["out"])
    if resume:
        if json.loads((root / "suite_config.json").read_text()) != config:
            raise ValueError("Suite configuration changed; use a new output directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "suite_config.json", config)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def command(label, arguments):
        cmd = [sys.executable, "-u", "-m", "genome_mining.rrc_cli", *map(str, arguments)]
        print(json.dumps({"stage": label, "command": cmd}), flush=True)
        with (logs / (label + ".log")).open("a", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        return result.returncode

    source = "--hf-model" if config.get("hf_model") else "--model"
    common = [source, config.get("hf_model") or config["model"], "--key-file", config["key_file"],
              "--device", config["device"], "--context-bases", config["context_bases"]]
    if config.get("trust_remote_code"):
        common += ["--trust-remote-code"]
    summary, failures = [], []
    for bits in config["bits"]:
        dataset = root / f"bits_{bits}" / "dataset"
        build = ["build", *common, "--contexts", config["contexts"], "--bits", bits,
                 "--length", config["length"], "--out", dataset]
        for field in ("precision_bits", "max_bases"):
            build += ["--" + field.replace("_", "-"), config[field]]
        if resume and dataset.exists():
            build += ["--resume"]
        code = command(f"bits_{bits}_build", build)
        run_path = dataset / "run.json"
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        if code:
            if run.get("status") != "complete_with_failures":
                raise RuntimeError(f"Generation failed; see {logs / f'bits_{bits}_build.log'}")
            failures.append({"bits": bits, "failed": run["failed"], "failure_rate": run["failure_rate"]})
            atomic_json(root / "capacity_failures.json", failures)
            continue
        if command(f"bits_{bits}_verify", ["verify", *common, "--contexts", config["contexts"], "--dataset", dataset]):
            raise RuntimeError("Independent verification failed; inspect logs")
        for detector in config["detectors"]:
            # Preserve interrupted evaluations; never erase partial evidence on resume.
            attempt = 1
            while True:
                evaluation = dataset.parent / f"{detector}_attempt_{attempt}"
                status = evaluation / "evaluation.json"
                if not evaluation.exists() or (status.exists() and json.loads(status.read_text()).get("status") == "complete"):
                    break
                attempt += 1
            if not evaluation.exists():
                args = ["evaluate", "--csv", dataset / "sequences.csv", "--out", evaluation,
                        "--detector", detector, "--device", config["device"]]
                for field in ("epochs", "batch_size", "seed", "bootstrap", "fpr"):
                    args += ["--" + field.replace("_", "-"), config[field]]
                if detector == "hyena-probe":
                    args += ["--hf-model", config["detector_hf_model"]]
                    if config.get("trust_remote_code"):
                        args += ["--trust-remote-code"]
                if command(f"bits_{bits}_{detector}_{attempt}", args):
                    raise RuntimeError("Evaluation failed; inspect logs, then --resume")
            report = json.loads((evaluation / "report.json").read_text())
            for task, values in report.items():
                summary.append({"bits": bits, "bits_per_base": bits/config["length"],
                                "detector": detector, "task": task, "auroc": values["auroc"],
                                "test_fpr": values["test_fpr"], "test_tpr": values["test_tpr"],
                                "failure_rate": run["failure_rate"], "report": str(evaluation / "report.json")})
            write_csv(root / "summary.csv", summary)
    atomic_json(root / "capacity_failures.json", failures)
    atomic_json(root / "suite_status.json", {"status": "complete_with_failures" if failures else "complete",
                                            "evaluated_tasks": len(summary), "failed_rates": failures})
    if failures:
        raise RuntimeError("Capacity failures recorded; those rates were NOT silently filtered/evaluated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_suite(resolve_config(args.config), args.resume)


if __name__ == "__main__":
    main()
