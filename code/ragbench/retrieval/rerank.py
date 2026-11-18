"""Cross-encoder reranking for the HYBRID+rerank BASELINE (2026-09-25).

A comparison arm only: HYBRID's RRF candidates (top RERANK_DEFAULTS["candidates"]) are re-scored by a multilingual
cross-encoder (BAAI/bge-reranker-v2-m3) on (question, chunk text) and re-ordered; nothing else changes (same evidence
packing, budget and shared answer call). LEX / LEX3 / AGENTIC* must NOT import this module - their comparison against
the reranked hybrid is the point of the arm.

    from ragbench.retrieval import rerank as R
    rr = R.get_reranker()                       # lazy, process-wide; loads the model on first use
    hits2, info, scores = R.rerank_hits(question, hits, rr, keep=40)

The model is not an Ollama model and makes no ledgered call: its cost is reported per question in trace["rerank"]
(candidates, ms, device, fp16, fallback). Device "auto": CUDA (fp16) when a GPU is available with at least
min_free_gb free, else CPU (fp32); a CUDA out-of-memory error falls back to CPU for the rest of the process and is
recorded (fallback). The model cache is HF_HOME (default <project>/work/models/hf when unset).
Tests inject any object with .score(question, texts) -> (list[float], info dict) (see FakeReranker in the tests).
"""
import os
import time
from pathlib import Path

RERANK_VERSION = "rerank-v1"
RERANK_DEFAULTS = {
    "model": "BAAI/bge-reranker-v2-m3",
    "candidates": 50,            # fused HYBRID hits re-scored (the rest of the fused list is not considered)
    "batch_size": 16,
    "max_length": 512,           # tokens per (question, chunk) pair; longer pairs are truncated (chunk side first)
    "device": "auto",            # auto | cuda | cpu
    "fp16": True,                # on CUDA only
    "min_free_gb": 1.5,          # auto: below this free GPU memory the CPU is used (the GPU is busy)
}

_PROJECT = Path(__file__).resolve().parents[2]


def default_hf_home():
    return os.environ.get("HF_HOME") or str(_PROJECT / "work" / "models" / "hf")


class CrossEncoderReranker:
    """transformers AutoModelForSequenceClassification scorer (the FlagEmbedding FlagReranker recipe: raw logit)."""

    def __init__(self, model=None, device=None, fp16=None, batch_size=None, max_length=None, min_free_gb=None):
        d = RERANK_DEFAULTS
        self.model_name = model or d["model"]
        self.want_device = device or d["device"]
        self.want_fp16 = d["fp16"] if fp16 is None else bool(fp16)
        self.batch_size = int(batch_size or d["batch_size"])
        self.max_length = int(max_length or d["max_length"])
        self.min_free_gb = float(d["min_free_gb"] if min_free_gb is None else min_free_gb)
        self.device = None
        self.fp16 = False
        self.fallback = None         # why the CPU is used when CUDA was wanted / possible
        self.load_ms = None
        self._tok = self._model = None

    # -- loading -----------------------------------------------------------------------------------------------------
    def _pick_device(self, torch):
        if self.want_device == "cpu":
            return "cpu", None
        if not torch.cuda.is_available():
            return "cpu", "cuda unavailable"
        if self.want_device == "auto":
            try:
                free, _total = torch.cuda.mem_get_info()
                if free / 2 ** 30 < self.min_free_gb:
                    return "cpu", "gpu busy: %.2f GB free < %.2f" % (free / 2 ** 30, self.min_free_gb)
            except Exception as e:  # noqa: BLE001
                return "cpu", "cuda mem_get_info failed: %s" % type(e).__name__
        return "cuda", None

    def load(self):
        if self._model is not None:
            return self
        os.environ.setdefault("HF_HOME", default_hf_home())
        t0 = time.perf_counter()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        dev, why = self._pick_device(torch)
        if dev == "cuda":
            try:
                model = model.half() if self.want_fp16 else model
                model = model.to("cuda")
                self.fp16 = self.want_fp16
            except RuntimeError as e:        # CUDA OOM while moving the weights
                why = "cuda load failed: %s" % str(e)[:120]
                model = model.float().to("cpu")
                dev, self.fp16 = "cpu", False
                _empty_cache(torch)
        self.device, self.fallback, self._model = dev, why, model
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return self

    def _to_cpu(self, why):
        torch = self._torch
        self._model = self._model.float().to("cpu")
        self.device, self.fp16, self.fallback = "cpu", False, why
        _empty_cache(torch)

    # -- scoring -----------------------------------------------------------------------------------------------------
    def _batch_scores(self, pairs):
        torch = self._torch
        enc = self._tok(pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        am = enc.get("attention_mask")
        if am is not None:                      # resource table: real (non-padding) vs padded input positions
            self._tok_nonpad += int(am.sum())
            self._tok_padded += int(am.numel())
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self._model(**enc, return_dict=True).logits.view(-1).float()
        return logits.cpu().tolist()

    def score(self, question, texts):
        """-> (scores: list[float] aligned with texts, info dict). Higher = more relevant (raw logit)."""
        self.load()
        t0 = time.perf_counter()
        self._tok_nonpad, self._tok_padded = 0, 0
        pairs = [[question, t] for t in texts]
        # sort by length so a batch pads to similar lengths (the output is put back in input order)
        order = sorted(range(len(pairs)), key=lambda i: len(texts[i]))
        out = [0.0] * len(pairs)
        oom = None
        i = 0
        while i < len(order):
            idx = order[i:i + self.batch_size]
            try:
                s = self._batch_scores([pairs[j] for j in idx])
            except RuntimeError as e:
                if self.device == "cuda" and "out of memory" in str(e).lower():
                    oom = "cuda out of memory at batch %d" % (i // self.batch_size)
                    self._to_cpu(oom)
                    continue                          # retry this batch on the CPU
                raise
            for j, v in zip(idx, s):
                out[j] = float(v)
            i += self.batch_size
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        info = {"model": self.model_name, "device": self.device, "fp16": self.fp16, "batch_size": self.batch_size,
                "max_length": self.max_length, "n": len(texts), "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "tokens_nonpad": self._tok_nonpad, "tokens_padded": self._tok_padded}
        if self.fallback:
            info["fallback"] = self.fallback
        if oom:
            info["oom_this_call"] = oom
        return out, info


def _empty_cache(torch):
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


_SHARED = {}


def get_reranker(**kw):
    """Process-wide reranker (one model load per process and settings)."""
    key = tuple(sorted(kw.items()))
    if key not in _SHARED:
        _SHARED[key] = CrossEncoderReranker(**kw)
    return _SHARED[key]


def rerank_hits(question, hits, reranker, keep=None, text_fn=None):
    """Re-order hits (best first) by the reranker's score of (question, text_fn(hit)); ties keep the input (RRF)
    order. -> (reranked list of the same objects, at most `keep`; info dict; the kept hits' scores).
    Every input hit is scored; `keep` only truncates the reranked list."""
    hits = list(hits)
    text_fn = text_fn or (lambda h: h.text)
    t0 = time.perf_counter()
    if hits:
        scores, info = reranker.score(question, [text_fn(h) for h in hits])
        if len(scores) != len(hits):
            raise ValueError("reranker returned %d scores for %d candidates" % (len(scores), len(hits)))
    else:
        scores, info = [], {"n": 0, "ms": 0.0}
    order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))
    kept = order[:keep] if keep is not None else order
    info = dict(info)
    info.update({"version": RERANK_VERSION, "candidates_reranked": len(hits), "kept": len(kept),
                 "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                 # [input (RRF) rank, reranker score] in the reranked order
                 "order": [[i, round(float(scores[i]), 4)] for i in order]})
    return [hits[i] for i in kept], info, [float(scores[i]) for i in kept]
