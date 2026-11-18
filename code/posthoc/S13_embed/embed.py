"""S13 GPU phase: multilingual-e5-large (HF, fp16 cuda) vectors of every chunk of the six test sets and the 156 questions.

    python embed.py probe      200 chunks, throughput + peak memory
    python embed.py encode     everything -> vectors/e5_chunks_<set>.npy (fp16, L2-normalised), vectors/e5_queries.npz

Texts: chunk = "passage: " + HybridIndex index_text(chunk) (the exact "<breadcrumb>\\n<chunk>" text bge-m3 embedded);
question = "query: " + normalize(question text) (the text HybridIndex.query_vector embeds). Mean pooling over the
attention mask, L2 normalisation (the model card's recipe), max_length 512 (the model's limit; longer texts truncated
and counted). No lock (COMMON.md GPU rule as updated): nvidia-smi checked before load; batch halves on CUDA OOM.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ.setdefault("HF_HOME", r"D:\KareOne\RAG Paper\work\models\hf")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

import s13lib as S  # noqa: E402  (imports S2a packs, which sets CUDA_VISIBLE_DEVICES=""; undone before CUDA init)

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MAXLEN = 512
E5_REV = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"   # HF commit downloaded 2026-11-10
LOG = S.HERE / "gpu_phase.json"


def nvsmi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def load():
    from huggingface_hub import snapshot_download
    path = str(Path(os.environ["HF_HOME"]) / "hub" / "models--intfloat--multilingual-e5-large" / "snapshots" / E5_REV)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to("cuda").eval()
    return tok, model, path


@torch.inference_mode()
def encode(tok, model, texts, bs):
    """Length-sorted batches; on OOM the batch size halves and the batch is retried. -> (fp32 L2 vectors, n_trunc, bs)"""
    lens = [len(x) for x in texts]
    order = np.argsort(lens)[::-1]
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    ntrunc = 0
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        batch = [texts[j] for j in idx]
        try:
            enc = tok(batch, max_length=MAXLEN, truncation=True, padding=True, return_tensors="pt",
                      return_overflowing_tokens=False)
            full = tok(batch, truncation=False)["input_ids"]
            enc = {k: v.to("cuda") for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1).float() / m.sum(1).float()
            v = torch.nn.functional.normalize(v, dim=-1)
            out[idx] = v.cpu().numpy()
            ntrunc += sum(len(f) > MAXLEN for f in full)
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print("OOM -> batch", bs, flush=True)
    return out, ntrunc, bs


def chunk_texts(set_name):
    return ["passage: " + S.H.index_text(c) for c in S.hx(set_name).chunks]


def main(mode):
    rec = {"mode": mode, "model": S.E5_MODEL, "nvidia_smi_before": nvsmi(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print("before load", rec["nvidia_smi_before"], flush=True)
    T0 = time.perf_counter()
    tok, model, path = load()
    rec["model_path"] = path
    rec["load_s"] = round(time.perf_counter() - T0, 1)
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode == "probe":
            texts = chunk_texts("C")[:100] + chunk_texts("BOOK_en")[:100]
            for bs in (16, 32):
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                v, nt, b = encode(tok, model, texts, bs)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                rec["probe_bs%d" % bs] = {"n": len(texts), "s": round(dt, 2), "chunks_per_s": round(len(texts) / dt, 1),
                                          "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                                          "truncated": nt, "nvidia_smi": nvsmi()}
                print(rec["probe_bs%d" % bs], flush=True)
            # sanity: an en question should prefer its own language passages; just print one cosine
            q = encode(tok, model, ["query: how much did revenue grow"], 1)[0][0]
            rec["probe_cos_example"] = float((v @ q).max())
        else:
            S.VEC.mkdir(exist_ok=True)
            bs = int(sys.argv[2]) if len(sys.argv) > 2 else 32
            rec["sets"] = {}
            t = time.perf_counter()
            Q = S.qlist()
            qtexts = ["query: " + S.normalize(q.text) for _s, _p, q in Q]
            qv, qt, _ = encode(tok, model, qtexts, 32)
            np.savez(S.VEC / "e5_queries.npz", qids=np.asarray([q.question_id for _s, _p, q in Q]),
                     split=np.asarray([s for s, _p, _q in Q]), vecs=qv.astype(np.float16))
            rec["queries"] = {"n": len(Q), "s": round(time.perf_counter() - t, 1), "truncated": qt}
            print(rec["queries"], flush=True)
            for sn in S.SETS:
                f = S.VEC / ("e5_chunks_%s.npy" % sn)
                if f.is_file():
                    rec["sets"][sn] = "cached"
                    continue
                t = time.perf_counter()
                texts = chunk_texts(sn)
                v, nt, bs = encode(tok, model, texts, bs)
                np.save(f, v.astype(np.float16))
                rec["sets"][sn] = {"n": len(texts), "s": round(time.perf_counter() - t, 1), "truncated": nt,
                                   "batch_end": bs}
                print(sn, rec["sets"][sn], nvsmi(), flush=True)
        rec["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        rec["nvidia_smi_end_loaded"] = nvsmi()
    finally:
        del model
        torch.cuda.empty_cache()
        rec["gpu_wall_s"] = round(time.perf_counter() - T0, 1)
        rec["nvidia_smi_after_unload"] = nvsmi()
        allr = json.loads(LOG.read_text(encoding="utf-8")) if LOG.is_file() else []
        allr.append(rec)
        S.wjson(LOG, allr)
        print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
