from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable


DNA_ALPHABET = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}


def reverse_complement(sequence: str) -> str:
    return str(sequence).upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


@runtime_checkable
class NucleotideProbabilityModel(Protocol):
    """Minimal next-base interface consumed by the reversible L5 coder."""

    model_id: str
    context_limit: int

    def probabilities(self, context: str) -> Sequence[float]:
        """Return probabilities ordered as A, C, G, and T."""

    def metadata(self) -> dict[str, object]:
        """Return enough information to audit the probability source."""


class KmerProbabilityModel:
    """Serializable variable-order Markov baseline for the L5 protocol.

    This is deliberately a reproducibility model, not a claim that short k-mers
    provide L5 naturalness. It validates the coding protocol locally and gives
    the A800 implementation a stable reference when a foundation model is added.
    """

    model_type = "kmer_next_base_v1"

    def __init__(
        self,
        *,
        order: int = 6,
        pseudocount: float = 0.5,
        minimum_context_count: int = 4,
        counts: dict[str, Sequence[int]] | None = None,
        trained_bases: int = 0,
        include_reverse_complement: bool = True,
    ) -> None:
        if not 0 <= order <= 12:
            raise ValueError("order must be between zero and twelve.")
        if pseudocount <= 0.0:
            raise ValueError("pseudocount must be positive.")
        if minimum_context_count < 0:
            raise ValueError("minimum_context_count cannot be negative.")
        self.order = int(order)
        self.context_limit = int(order)
        self.pseudocount = float(pseudocount)
        self.minimum_context_count = int(minimum_context_count)
        self.counts = {
            str(context): tuple(int(value) for value in values)
            for context, values in (counts or {}).items()
        }
        if any(len(values) != 4 or any(value < 0 for value in values) for values in self.counts.values()):
            raise ValueError("Every k-mer context needs four non-negative counts.")
        self.trained_bases = int(trained_bases)
        self.include_reverse_complement = bool(include_reverse_complement)
        self.model_id = self._fingerprint()

    @classmethod
    def fit(
        cls,
        sequences: Iterable[str],
        *,
        order: int = 6,
        pseudocount: float = 0.5,
        minimum_context_count: int = 4,
        include_reverse_complement: bool = True,
    ) -> "KmerProbabilityModel":
        mutable: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        trained_bases = 0

        def consume(raw_sequence: str) -> int:
            consumed = 0
            run = ""
            for base in str(raw_sequence).upper():
                if base not in BASE_TO_INDEX:
                    run = ""
                    continue
                max_length = min(order, len(run))
                mutable[""][BASE_TO_INDEX[base]] += 1
                for length in range(1, max_length + 1):
                    mutable[run[-length:]][BASE_TO_INDEX[base]] += 1
                run = (run + base)[-order:] if order else ""
                consumed += 1
            return consumed

        for sequence in sequences:
            normalized = str(sequence).upper()
            trained_bases += consume(normalized)
            if include_reverse_complement:
                consume(reverse_complement(normalized))

        return cls(
            order=order,
            pseudocount=pseudocount,
            minimum_context_count=minimum_context_count,
            counts={key: tuple(values) for key, values in mutable.items()},
            trained_bases=trained_bases,
            include_reverse_complement=include_reverse_complement,
        )

    @classmethod
    def from_fasta(
        cls,
        fasta_paths: Iterable[str | Path],
        *,
        order: int = 6,
        pseudocount: float = 0.5,
        minimum_context_count: int = 4,
        include_reverse_complement: bool = True,
        max_bases: int | None = None,
    ) -> "KmerProbabilityModel":
        from ..fasta import iter_fasta_records

        if max_bases is not None and max_bases <= 0:
            raise ValueError("max_bases must be positive when provided.")

        def sequences():
            remaining = max_bases
            for path in fasta_paths:
                for record in iter_fasta_records(path):
                    if remaining is None:
                        yield record.sequence
                        continue
                    if remaining <= 0:
                        return
                    chunk = record.sequence[:remaining]
                    if chunk:
                        yield chunk
                        remaining -= len(chunk)

        return cls.fit(
            sequences(),
            order=order,
            pseudocount=pseudocount,
            minimum_context_count=minimum_context_count,
            include_reverse_complement=include_reverse_complement,
        )

    def probabilities(self, context: str) -> tuple[float, float, float, float]:
        suffix = ""
        for base in reversed(str(context).upper()):
            if base not in BASE_TO_INDEX or len(suffix) >= self.order:
                break
            suffix = base + suffix

        selected = self.counts.get("", (0, 0, 0, 0))
        for length in range(min(self.order, len(suffix)), 0, -1):
            candidate = self.counts.get(suffix[-length:])
            if candidate is not None and sum(candidate) >= self.minimum_context_count:
                selected = candidate
                break
        values = [value + self.pseudocount for value in selected]
        total = sum(values)
        return tuple(value / total for value in values)  # type: ignore[return-value]

    def _serializable(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "order": self.order,
            "pseudocount": self.pseudocount,
            "minimum_context_count": self.minimum_context_count,
            "trained_bases": self.trained_bases,
            "include_reverse_complement": self.include_reverse_complement,
            "alphabet": DNA_ALPHABET,
            "counts": {key: list(self.counts[key]) for key in sorted(self.counts)},
        }

    def _fingerprint(self) -> str:
        canonical = json.dumps(
            self._serializable(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:16]

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_id": self.model_id,
            "order": self.order,
            "context_limit": self.context_limit,
            "pseudocount": self.pseudocount,
            "minimum_context_count": self.minimum_context_count,
            "trained_bases": self.trained_bases,
            "include_reverse_complement": self.include_reverse_complement,
            "n_contexts": len(self.counts),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = {**self._serializable(), "model_id": self.model_id}
        output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "KmerProbabilityModel":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported probability model type: {document.get('model_type')!r}")
        model = cls(
            order=int(document["order"]),
            pseudocount=float(document["pseudocount"]),
            minimum_context_count=int(document["minimum_context_count"]),
            counts=document["counts"],
            trained_bases=int(document.get("trained_bases", 0)),
            include_reverse_complement=bool(document.get("include_reverse_complement", True)),
        )
        declared = str(document.get("model_id", ""))
        if declared and declared != model.model_id:
            raise ValueError(f"Probability model fingerprint mismatch: file={declared}, actual={model.model_id}")
        return model


class HuggingFaceCausalDNAProbabilityModel:
    """Experimental adapter for checkpoints that expose next-token logits.

    The current HyenaDNA locator uses a representation backbone. Not every
    HyenaDNA Hugging Face checkpoint exposes a causal-LM head, so this adapter
    fails explicitly when logits are unavailable instead of silently treating
    embeddings as probabilities.
    """

    token_ids = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cuda",
        context_limit: int = 8192,
        local_files_only: bool = False,
        logit_rounding_digits: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "A Hugging Face causal DNA model requires torch and transformers."
            ) from exc
        kwargs = {"trust_remote_code": True, "local_files_only": bool(local_files_only)}
        if revision:
            kwargs["revision"] = revision
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        self.model.eval()
        self.model_name = str(model_name)
        self.revision = revision
        self.device = str(device)
        self.context_limit = int(context_limit)
        self.logit_rounding_digits = int(logit_rounding_digits)
        self.model_id = hashlib.sha256(
            f"{self.model_name}|{self.revision or ''}|{self.context_limit}".encode("utf-8")
        ).hexdigest()[:16]

    def probabilities(self, context: str) -> tuple[float, float, float, float]:
        normalized = "".join(base if base in self.token_ids else "N" for base in str(context).upper())
        normalized = normalized[-self.context_limit :] or "N"
        ids = [self.token_ids[base] for base in normalized]
        input_ids = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            output = self.model(input_ids=input_ids, return_dict=True)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise ValueError(
                f"Checkpoint {self.model_name!r} has no causal-LM logits. "
                "Use a checkpoint with AutoModelForCausalLM support or a custom adapter."
            )
        base_ids = [self.token_ids[base] for base in DNA_ALPHABET]
        values = logits[0, -1, base_ids].to(self.torch.float64).cpu()
        values = self.torch.round(values * (10**self.logit_rounding_digits)) / (
            10**self.logit_rounding_digits
        )
        probabilities = self.torch.softmax(values, dim=-1).tolist()
        return tuple(float(value) for value in probabilities)  # type: ignore[return-value]

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "huggingface_causal_dna_v1",
            "model_id": self.model_id,
            "model_name": self.model_name,
            "revision": self.revision,
            "context_limit": self.context_limit,
            "logit_rounding_digits": self.logit_rounding_digits,
            "warning": "Cross-device reproducibility must be verified before formal decoding experiments.",
        }


def load_probability_model(path: str | Path) -> NucleotideProbabilityModel:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = document.get("model_type")
    if model_type == KmerProbabilityModel.model_type:
        return KmerProbabilityModel.load(path)
    raise ValueError(f"Unsupported serialized probability model type: {model_type!r}")
