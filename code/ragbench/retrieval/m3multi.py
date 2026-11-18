"""bge-m3 multi-functionality encoder: dense + sparse (lexical weights) + multi-vector (ColBERT) in ONE forward pass.

A bi-encoder, NOT a cross-encoder: queries and chunks are encoded independently, so every chunk-side representation is
built at INDEX time (offline) and only the question is encoded at query time. Re-implements the BGEM3FlagModel
inference recipe (FlagEmbedding) on plain transformers so no extra package is needed:

    dense   = normalize(h[:, 0])                                   (CLS)
    sparse  = {token id: max relu(sparse_linear(h_t))}             special tokens (cls/eos/pad/unk) dropped
    colbert = normalize(colbert_linear(h[:, 1:]) * mask[:, 1:])    one row per non-CLS token (incl. </s>)

Scores (Chen et al. 2024, "BGE M3-Embedding", arXiv 2402.03216):
    s_dense   = q . p
    s_lex     = sum over shared tokens of w_q(t) * w_p(t)
    s_colbert = mean over query rows of max over passage rows of q_i . p_j
    s_all     = w1 s_dense + w2 s_lex + w3 s_colbert   (paper: 1 : 0.3 : 1)

Model weights: BAAI/bge-m3 (pytorch_model.bin + colbert_linear.pt + sparse_linear.pt) in HF_HOME
(default <project>/work/models/hf). Device auto: CUDA fp16 when available with min_free_gb free, else CPU fp32.
"""
import os
import time
from pathlib import Path

import numpy as np

M3_VERSION = "m3multi-v1"
M3_DEFAULTS = {"model": "BAAI/bge-m3", "max_length": 512, "query_max_length": 128, "batch_size": 16,
               "device": "auto", "fp16": True, "min_free_gb": 1.5, "weights": (1.0, 0.3, 1.0)}
_PROJECT = Path(__file__).resolve().parents[2]


def default_hf_home():
    """HF_HOME, else <project>/work/models/hf (a worktree has no work/: then the config's project_root is tried)."""
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    p = _PROJECT / "work" / "models" / "hf"
    try:
        from ..config import load_config
        q = Path(load_config().project_root) / "work" / "models" / "hf"
        if q.exists():
            p = q
    except Exception:  # noqa: BLE001
        pass
    return str(p)


class M3Encoder:
    def __init__(self, model=None, device=None, fp16=None, min_free_gb=None):
        d = M3_DEFAULTS
        self.model_name = model or d["model"]
        self.want_device = device or d["device"]
        self.want_fp16 = d["fp16"] if fp16 is None else bool(fp16)
        self.min_free_gb = float(d["min_free_gb"] if min_free_gb is None else min_free_gb)
        self.device = None
        self.fp16 = False
        self._model = None

    def load(self):
        if self._model is not None:
            return self
        os.environ.setdefault("HF_HOME", default_hf_home())
        t0 = time.perf_counter()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        pats = ["config.json", "pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]
        snaps = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + self.model_name.replace("/", "--")) / "snapshots"
        local = [p for p in (snaps.glob("*") if snaps.exists() else [])
                 if all((p / f).exists() for f in pats)]         # a complete local copy: never touch the network
        path = str(local[0]) if local else snapshot_download(self.model_name, allow_patterns=pats)
        self.revision = Path(path).name              # HF snapshot commit (artifact identity, LEX8)
        self.tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        hid = model.config.hidden_size
        self.colbert = torch.nn.Linear(hid, hid)
        self.colbert.load_state_dict(torch.load(os.path.join(path, "colbert_linear.pt"), map_location="cpu"))
        self.sparse = torch.nn.Linear(hid, 1)
        self.sparse.load_state_dict(torch.load(os.path.join(path, "sparse_linear.pt"), map_location="cpu"))
        dev = "cpu"
        if self.want_device != "cpu" and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if self.want_device == "cuda" or free >= self.min_free_gb:
                dev = "cuda"
        mods = [model, self.colbert, self.sparse]
        for m in mods:
            m.eval()
            if dev == "cuda":
                m.to("cuda")
                if self.want_fp16:
                    m.half()
        self.fp16 = dev == "cuda" and self.want_fp16
        self.device, self._model = dev, model
        self.unused = {i for i in (self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id,
                                   self.tok.unk_token_id) if i is not None}
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return self

    def encode(self, texts, max_length=None, batch_size=None, colbert=True):
        """-> list of dicts {dense: f32[1024], sparse: {id: w}, colbert: f16[n, 1024] | None, tokens: n}."""
        self.load()
        torch = self.torch
        max_length = int(max_length or M3_DEFAULTS["max_length"])
        bs = int(batch_size or M3_DEFAULTS["batch_size"])
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                h = self._model(**enc, return_dict=True).last_hidden_state
                dense = torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu().numpy()
                sw = torch.relu(self.sparse(h)).squeeze(-1).float().cpu().numpy()
                cv = None
                if colbert:
                    cv = self.colbert(h[:, 1:]) * enc["attention_mask"][:, 1:][:, :, None]
                    cv = torch.nn.functional.normalize(cv.float(), dim=-1).half().cpu().numpy()
            ids = enc["input_ids"].cpu().numpy()
            am = enc["attention_mask"].cpu().numpy()
            for b in range(ids.shape[0]):
                n = int(am[b].sum())
                sp = {}
                for t, w in zip(ids[b, :n], sw[b, :n]):
                    t = int(t)
                    if t in self.unused or w <= 0:
                        continue
                    if w > sp.get(t, 0.0):
                        sp[t] = float(w)
                out.append({"dense": dense[b], "sparse": sp, "tokens": n,
                            "colbert": (cv[b, :n - 1] if cv is not None else None)})
        return out


def sparse_score(qs, ps):
    """lexical-matching score of two {token id: weight} dicts."""
    if len(qs) > len(ps):
        qs, ps = ps, qs
    return float(sum(w * ps.get(t, 0.0) for t, w in qs.items()))


def colbert_score(qv, pv):
    """MaxSim: mean over query rows of the best passage row (both row-normalized)."""
    if qv is None or pv is None or len(qv) == 0 or len(pv) == 0:
        return 0.0
    s = qv.astype(np.float32) @ pv.astype(np.float32).T
    return float(s.max(axis=1).mean())


class SparseMatrix:
    """Chunk-side lexical weights as one CSR matrix (rows = chunks) for a whole-pool sparse channel in one product."""

    def __init__(self, dicts, vocab_size):
        from scipy.sparse import csr_matrix
        indptr, idx, val = [0], [], []
        for d in dicts:
            for t, w in d.items():
                idx.append(t)
                val.append(w)
            indptr.append(len(idx))
        self.M = csr_matrix((np.asarray(val, np.float32), np.asarray(idx, np.int32), np.asarray(indptr, np.int64)),
                            shape=(len(dicts), vocab_size))

    def scores(self, q):
        qv = np.zeros(self.M.shape[1], np.float32)
        for t, w in q.items():
            qv[t] = w
        return np.asarray(self.M @ qv).ravel()
