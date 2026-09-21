"""Offline probability sources for the RRC experiment."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .model_adapter import load_probability_model


class LocalCausalDNA:
    """Single-base HF tokenizer + causal head; fail on incompatible checkpoints.

    Both directions use batch-one full-prefix calls, not mixed cache/teacher
    forcing paths. Remote model code runs only with explicit user opt-in.
    """

    def __init__(self, path, *, device="cpu", context_limit=1024, trust_remote_code=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(path).resolve()
        if not root.is_dir():
            raise ValueError("HF model must be a downloaded local directory")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True,
                                                  trust_remote_code=trust_remote_code)
        self.ids = {}
        for base in "ACGTN":
            ids = tokenizer.encode(base, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer must encode {base} as one non-UNK token")
            self.ids[base] = ids[0]
        if len(set(self.ids.values())) != 5:
            raise ValueError("Base tokens must be distinct")
        if tokenizer.encode("ACGTNAC", add_special_tokens=False) != [self.ids[b] for b in "ACGTNAC"]:
            raise ValueError("Context-dependent/k-mer tokenizer is unsupported")
        self.model = AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=trust_remote_code,
        ).to(device).eval()
        limit = getattr(self.model.config, "max_seq_len", None)
        if limit is not None and context_limit > limit:
            raise ValueError(f"context_limit={context_limit} exceeds checkpoint limit={limit}")
        self.torch, self.device, self.context_limit = torch, device, context_limit
        digest = hashlib.sha256()
        for file in sorted(root.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".py", ".bin", ".safetensors", ".pt", ".ckpt"}:
                digest.update(file.relative_to(root).as_posix().encode())
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        self.model_id = digest.hexdigest()

    def metadata(self):
        return {"model_type": "rrc_local_hf_causal_v1", "model_id": self.model_id,
                "token_ids": self.ids, "context_limit": self.context_limit,
                "torch_version": str(self.torch.__version__), "forward": "batch1_full_prefix"}

    def probabilities(self, context):
        text = context[-self.context_limit:] or "N"
        tokens = self.torch.tensor([[self.ids[b] for b in text]], device=self.device)
        with self.torch.inference_mode():
            out = self.model(input_ids=tokens)
        logits = getattr(out, "logits", None)
        if logits is None or logits.ndim != 3 or logits.shape[1] != tokens.shape[1]:
            raise ValueError("Checkpoint must expose per-position causal LM logits")
        values = logits[0, -1, [self.ids[b] for b in "ACGT"]].double().cpu()
        return self.torch.softmax(values, dim=-1).tolist()


def probability_source(args):
    if args.model:
        return load_probability_model(args.model)
    return LocalCausalDNA(args.hf_model, device=args.device, context_limit=args.context_bases,
                         trust_remote_code=args.trust_remote_code)
