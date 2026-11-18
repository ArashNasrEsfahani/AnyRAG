"""HYBRID+rerank smoke: retrieval only (no answer call), 5 Persian (C) + 5 English (D) dev questions.

Run:  python -m ragbench.tools.rerank_smoke [--cpu-only | --gpu-only] [--n 5]   (outputs to $RERANK_SMOKE_OUT)
Measures rerank latency on GPU (auto device) and CPU (2 threads), and dumps the reranked top-5 for eyeballing.
Only the question is embedded (Ollama bge-m3, like HYBRID); no chat call is made (generate_answer is stubbed).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
OUT = Path(os.environ.get("RERANK_SMOKE_OUT", r"D:\KareOne\RAG Paper\work\upgrades\rerank"))
os.environ.setdefault("HF_HOME", r"D:\KareOne\RAG Paper\work\models\hf")

from ragbench.approaches import get_approach  # noqa: E402
from ragbench.approaches import hybrid as HY  # noqa: E402
from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import load_questions  # noqa: E402
from ragbench.llm import Ledger, OllamaClient  # noqa: E402
from ragbench.pools import load_pool  # noqa: E402
from ragbench.retrieval import rerank as R  # noqa: E402

MD = Path(r"D:\KareOne\RAG Paper\work\multidoc")
HY.HYBRID_DEFAULTS["embed_at_build"] = False      # never embed chunks here: use the prebuilt shared index as is


def gold_rank(hits_units, gold):
    for r, us in enumerate(hits_units):
        if set(us) & gold:
            return r
    return None


def run_pass(label, reranker, qs, cfg, client, n):
    rows = []
    aps = {}
    for corpus, setting, pool_file in (("C", "fa-fa", "C-full.json"), ("D", "en-en", "D-full.json")):
        ap = get_approach("HYBRID", cfg=cfg, client=client, flags={"rerank": True})
        ap.reranker = reranker
        ap.pool = load_pool(MD / "pools" / pool_file)
        ap.variants_version = "variants-v2"
        ap.generate_answer = lambda query, ev, purpose="answer": "(dry-run: no answer call)"
        aps[corpus] = ap
        for q in qs[corpus][:n]:
            t0 = time.perf_counter()
            rec = ap.answer(q, setting)
            wall = (time.perf_counter() - t0) * 1000.0
            tr = rec.trace or {}
            ri = tr.get("rerank", {})
            lang = setting[:2]
            gold = set(q.gold_units.get(lang) or [])
            after = [u["unit_ids"] for u in rec.retrieved_units]
            rrf_order = sorted(rec.retrieved_units, key=lambda u: u["meta"]["rrf_rank"])
            row = {"pass": label, "qid": q.id, "setting": setting, "error": rec.error,
                   "question": q.question_fa if lang == "fa" else q.question_en,
                   "candidates_reranked": ri.get("candidates_reranked"), "rerank_ms": ri.get("ms"),
                   "device": ri.get("device"), "fp16": ri.get("fp16"), "fallback": ri.get("fallback"),
                   "load_ms": ri.get("load_ms"), "retrieval_ms": tr.get("retrieval_ms"), "wall_ms": round(wall, 1),
                   "gold_n": len(gold), "gold_first_rank_after": gold_rank(after, gold),
                   "gold_first_rank_rrf_among_kept": gold_rank([u["unit_ids"] for u in rrf_order], gold),
                   "gold_hits_top10_after": sum(1 for u in rec.retrieved_units[:10] if set(u["unit_ids"]) & gold),
                   "gold_hits_top10_rrf": sum(1 for u in sorted(rec.retrieved_units, key=lambda u: u["meta"]["rrf_rank"])[:10]
                                              if set(u["unit_ids"]) & gold),
                   "top5": [{"rerank": u["meta"]["rerank"], "rrf_rank": u["meta"]["rrf_rank"], "doc": u["doc_id"],
                             "gold": bool(set(u["unit_ids"]) & gold), "text": u["text"][:220].replace("\n", " ")}
                            for u in rec.retrieved_units[:5]],
                   "rrf_top3": [{"rrf_rank": u["meta"]["rrf_rank"], "rerank": u["meta"]["rerank"], "doc": u["doc_id"],
                                 "gold": bool(set(u["unit_ids"]) & gold), "text": u["text"][:160].replace("\n", " ")}
                                for u in rrf_order[:3]]}
            if rec.error:
                row["traceback"] = (tr or {}).get("traceback")
            rows.append(row)
            print("%s %s %s dev=%s n=%s rerank_ms=%s gold@after=%s gold@rrf=%s err=%s" % (
                label, q.id, setting, row["device"], row["candidates_reranked"], row["rerank_ms"],
                row["gold_first_rank_after"], row["gold_first_rank_rrf_among_kept"], rec.error), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    client = OllamaClient(cfg, Ledger(path=OUT / "smoke_ledger.jsonl"))
    qs = {"C": load_questions(MD / "dev" / "dev_v1_C.jsonl"), "D": load_questions(MD / "dev" / "dev_v1_D.jsonl")}
    import torch
    torch.set_num_threads(2)
    env = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "threads": torch.get_num_threads(), "hf_home": os.environ["HF_HOME"]}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    all_rows = []
    if not a.cpu_only:
        gpu = R.CrossEncoderReranker(device="auto").load()
        env["gpu_pass_device"], env["gpu_pass_fallback"], env["gpu_load_ms"] = gpu.device, gpu.fallback, gpu.load_ms
        if torch.cuda.is_available():
            env["gpu_mem_alloc_mb_after_load"] = round(torch.cuda.memory_allocated() / 2 ** 20)
        all_rows += run_pass("gpu", gpu, qs, cfg, client, a.n)
        if torch.cuda.is_available():
            env["gpu_max_mem_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        del gpu
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if not a.gpu_only:
        cpu = R.CrossEncoderReranker(device="cpu").load()
        env["cpu_load_ms"] = cpu.load_ms
        all_rows += run_pass("cpu", cpu, qs, cfg, client, a.n)
    (OUT / "smoke_results.json").write_text(json.dumps({"env": env, "rows": all_rows}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    for p in ("gpu", "cpu"):
        ms = [r["rerank_ms"] for r in all_rows if r["pass"] == p and r["rerank_ms"] is not None]
        if ms:
            ms_s = sorted(ms)
            print("%s: n=%d rerank ms mean %.0f median %.0f max %.0f" % (p, len(ms), sum(ms) / len(ms), ms_s[len(ms) // 2], max(ms)))
    print(json.dumps(env))


if __name__ == "__main__":
    main()
