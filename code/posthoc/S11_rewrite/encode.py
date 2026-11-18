"""S11 step 2 (the one GPU phase per split): embed the original questions, the HyDE passages and the multi-query
rewrites through the frozen query path (OllamaClient.embed([normalize(text)]), one text per call, exactly as
HybridIndex.query_vector does at query time). -> vectors_<split>.npz (keys, vecs) and gpu_phases.jsonl.
Also compares the re-encoded original questions with S9's align_vectors.npz (q_oll, the same path, 2026-11-10).
    python encode.py dev|test"""
import datetime as dt
import json
import subprocess
import sys
import time

import numpy as np

import s11lib as L
from ragbench.llm import Ledger, OllamaClient
from ragbench.normalize import normalize


def nvsmi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    split = sys.argv[1]
    rws = {(r["qid"], r["method"]): r for r in L.jl(L.HERE / ("rewrites_%s.jsonl" % split)) if r.get("ok")}
    qids = sorted({q for q, _ in rws})
    Q = {q.question_id: q for q in L.queries(split, set(qids))}
    texts = []                                    # (key, text)
    for qid in qids:
        texts.append(("%s|ORIG|0" % qid, Q[qid].text))
        if (qid, "HYDE") in rws:
            texts.append(("%s|HYDE|0" % qid, rws[(qid, "HYDE")]["rewrites"][0]))
        if (qid, "MQ") in rws:
            for j, t in enumerate(rws[(qid, "MQ")]["rewrites"]):
                texts.append(("%s|MQ|%d" % (qid, j), t))
    out = L.HERE / ("vectors_%s.npz" % split)
    before = nvsmi()
    print("texts", len(texts), "nvidia-smi before", before, flush=True)
    oc = OllamaClient(L.cfg, ledger=Ledger(path=str(L.HERE / "ollama_ledger.jsonl")))
    t0 = time.perf_counter()
    vecs = []
    with oc.ledger.scope(phase="query", approach="S11-encode-" + split):
        for k, t in texts:
            er = oc.embed([normalize(t or "")], purpose="query-embed")
            vecs.append(np.asarray(er.vectors[0], dtype=np.float32))
    wall = time.perf_counter() - t0
    V = np.stack(vecs)
    np.savez(out, keys=np.asarray([k for k, _ in texts]), vecs=V)
    after = nvsmi()
    # check vs S9's re-encoded questions (same Ollama path)
    z = np.load(L.WORK / "posthoc_2026-11" / "S9_gpu" / "align_vectors.npz")
    s9 = {str(q): v for q, v in zip(z["qids"], z["q_oll"])}
    eq, cos = 0, []
    for (k, _t), v in zip(texts, vecs):
        qid, m, _ = k.split("|")
        if m == "ORIG" and qid in s9:
            a, b = L.l2(v), L.l2(s9[qid])
            eq += int(np.array_equal(a, b))
            cos.append(float(a @ b))
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "split": split, "n_texts": len(texts),
           "n_questions": len(qids), "wall_s": round(wall, 2), "nvidia_smi_before": before,
           "nvidia_smi_after": after, "orig_vs_S9_n": len(cos), "orig_vs_S9_bit_equal": eq,
           "orig_vs_S9_cos_min": (round(min(cos), 7) if cos else None),
           "embed_model": L.cfg.models["embed"], "embed_digest": L.cfg.models.get("embed_digest")}
    with open(L.HERE / "gpu_phases.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
