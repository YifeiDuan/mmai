"""Qwen2.5-VL-3B-Instruct text-only LLM proposer for inverse design candidates.

The proposer asks an instruction-tuned LLM to suggest new ABO3 perovskite
candidates given a target band gap and a history of past proposals + their
predicted band gaps. Output is parsed as a strict JSON list of
``{"A", "B", "base"}`` objects, then filtered against allowed A/B element
vocabularies and known base material IDs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

logger = logging.getLogger(__name__)

Candidate = Tuple[str, str, str]  # (A, B, base_material_id)


_JSON_LIST_RE = re.compile(r"\[\s*\{.*?\}\s*\]", flags=re.DOTALL)


def parse_proposals(
    raw_text: str,
    a_vocab: Iterable[str],
    b_vocab: Iterable[str],
    base_set: Iterable[str],
) -> List[Candidate]:
    """Parse the LLM's reply into a filtered list of valid candidates.

    The model is asked to emit a JSON list of objects with keys ``A``, ``B``,
    ``base``. We extract the FIRST ``[ ... ]`` block in the reply, ``json.loads``
    it, then drop any candidate whose A is not in ``a_vocab``, B not in
    ``b_vocab``, or base not in ``base_set``. Returns ``[]`` on any failure
    (no JSON found, JSON malformed, all rows filtered out).
    """
    a_set = set(a_vocab)
    b_set = set(b_vocab)
    base_set = set(base_set)

    match = _JSON_LIST_RE.search(raw_text)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    out: List[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        a = item.get("A")
        b = item.get("B")
        base = item.get("base")
        if not (isinstance(a, str) and isinstance(b, str) and isinstance(base, str)):
            continue
        if a not in a_set or b not in b_set or base not in base_set:
            continue
        out.append((a, b, base))
    return out


def random_fallback_propose(
    n: int,
    a_vocab: Sequence[str],
    b_vocab: Sequence[str],
    base_set: Sequence[str],
    rng: Optional[random.Random] = None,
) -> List[Candidate]:
    """Uniform random sample of n (A, B, base) tuples from the vocabs.

    Used when the LLM fails to produce a parseable proposal twice in a row.
    Each axis is sampled independently with replacement; we don't dedupe across
    proposals because the agent's outer loop will dedupe by candidate identity.
    """
    rng = rng or random
    out: List[Candidate] = []
    a_list = list(a_vocab)
    b_list = list(b_vocab)
    base_list = list(base_set)
    for _ in range(n):
        out.append((rng.choice(a_list), rng.choice(b_list), rng.choice(base_list)))
    return out


def _format_history(history: Sequence[dict], max_iters_to_show: int = 3) -> str:
    """Render the last ``max_iters_to_show`` iterations as a readable bullet list."""
    if not history:
        return "(no prior proposals)"
    last = list(history)[-max_iters_to_show:]
    lines = []
    for h in last:
        it = h.get("iter")
        cands = h.get("candidates", [])
        for c in cands:
            lines.append(
                f"  iter {it}: A={c['A']} B={c['B']} base={c['base_mid']} "
                f"-> pred_bg={c['pred_bg_mean']:.3f} eV "
                f"(std={c['pred_bg_std']:.3f}, |err|={c['error_to_target']:.3f})"
            )
    return "\n".join(lines) if lines else "(no prior proposals)"


def build_prompt(
    target_bg: float,
    history: Sequence[dict],
    a_vocab: Sequence[str],
    b_vocab: Sequence[str],
    base_sample: Sequence[str],
    n_proposals: int,
) -> str:
    """Construct the user-message prompt sent to the LLM."""
    a_str = ", ".join(sorted(a_vocab))
    b_str = ", ".join(sorted(b_vocab))
    base_str = ", ".join(base_sample)
    history_str = _format_history(history)
    return (
        f"You are a materials scientist designing ABO3 perovskite oxides. "
        f"The target band gap is {target_bg:.2f} eV.\n\n"
        f"Allowed A-site element symbols: {a_str}\n"
        f"Allowed B-site element symbols: {b_str}\n"
        f"Allowed base structures (Materials Project IDs you may template from): "
        f"{base_str}\n\n"
        f"History of previous proposals and their forward-model predictions:\n"
        f"{history_str}\n\n"
        f"Propose {n_proposals} new ABO3 candidates likely to match a band gap of "
        f"{target_bg:.2f} eV. Each candidate must specify an A-site element, a "
        f"B-site element, and a base material_id from the lists above. Avoid "
        f"repeating prior proposals.\n\n"
        f"Respond with a single JSON array on the FIRST line, then a brief "
        f"one-sentence rationale per candidate. Use this exact JSON schema:\n"
        f'[{{"A": "Sr", "B": "Ti", "base": "mp-XXXX"}}, ...]'
    )


class QwenProposer:
    """Wraps a Qwen2.5-VL-3B-Instruct model in text-only chat mode.

    Lazy-loads the model on first ``propose`` call to keep import-time cheap
    for unit tests of the parser. After two consecutive parse failures the
    proposer falls back to ``random_fallback_propose`` and emits a warning.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    DEFAULT_FALLBACK = "Qwen/Qwen2.5-3B-Instruct"
    SYSTEM_PROMPT = (
        "You are an expert solid-state chemist. Output exactly the requested "
        "JSON schema with no additional formatting outside the array."
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 0,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._model = None
        self._processor = None
        self._is_vl = "VL" in model_name
        self._rng = random.Random(seed)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if self._is_vl:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map if torch.cuda.is_available() else None,
            )
        else:
            from transformers import AutoModelForCausalLM
            self._processor = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map if torch.cuda.is_available() else None,
            )
        self._model.eval()

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch
        self._ensure_loaded()
        if self._is_vl:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=[text], return_tensors="pt")
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        if self._is_vl:
            full = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        else:
            full = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        # Strip echoed prompt: keep only the assistant tail.
        parts = full.split("assistant\n")
        return parts[-1].strip() if len(parts) > 1 else full.strip()

    def propose(
        self,
        target_bg: float,
        history: Sequence[dict],
        a_vocab: Sequence[str],
        b_vocab: Sequence[str],
        base_set: Sequence[str],
        n_proposals: int = 5,
        max_base_sample: int = 20,
    ) -> List[Candidate]:
        """Ask the LLM for ``n_proposals`` new ABO3 candidates.

        Tries up to 2 generations; on consecutive parse failures falls back
        to uniform random sampling and logs a warning.
        """
        base_sample = list(base_set)
        if len(base_sample) > max_base_sample:
            base_sample = self._rng.sample(base_sample, max_base_sample)

        prompt = build_prompt(
            target_bg=target_bg, history=history,
            a_vocab=a_vocab, b_vocab=b_vocab, base_sample=base_sample,
            n_proposals=n_proposals,
        )

        for attempt in range(2):
            try:
                raw = self._generate(self.SYSTEM_PROMPT, prompt)
            except Exception:
                logger.exception("Qwen generation crashed (attempt %d)", attempt)
                continue
            cands = parse_proposals(raw, a_vocab, b_vocab, base_set)
            if cands:
                return cands[:n_proposals]
            logger.warning(
                "Empty/unparseable proposal on attempt %d. Raw head=%r",
                attempt, raw[:200],
            )

        logger.warning("LLM failed twice; falling back to random sampling.")
        return random_fallback_propose(
            n=n_proposals, a_vocab=a_vocab, b_vocab=b_vocab,
            base_set=base_set, rng=self._rng,
        )
