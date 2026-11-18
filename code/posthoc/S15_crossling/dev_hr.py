"""Dev smoke for HR_6000_TRANS: build the HR 6,000 packs of the 5 dev translations (Ollama embed + GPU rerank)
-> dev_packs/HR_6000_trans/<qid>.json; then `python answer.py --split dev --arms HR_6000_TRANS`."""
import json
import os

os.environ["HF_HOME"] = r"D:\KareOne\RAG Paper\work\models\hf"
os.environ["HF_HUB_OFFLINE"] = "1"
import s15lib as L  # noqa: E402

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
from ragbench.llm import Ledger, OllamaClient  # noqa: E402
from ragbench.retrieval import rerank as R  # noqa: E402

L.CFG.ollama["cross_process_lock"] = False
oc = OllamaClient(L.CFG, ledger=Ledger(path=str(L.HERE / "ollama_ledger.jsonl")))
rr = R.get_reranker().load()
d = L.HERE / "dev_packs" / "HR_6000_trans"
d.mkdir(parents=True, exist_ok=True)
for r in L.jl(L.HERE / "dev_translations.jsonl"):
    if r["direction"] != "forward" or r.get("error"):
        continue
    q = L.swapped_query(r["qid"], r["translation"], "dev")
    qv = oc.embed([L.normalize(q.text)], purpose="query-embed", phase="smoke", approach="S15-dev").vectors[0]
    its, _f, _i = L.hr_list(q, qv, rr)
    ev, msgs, npred = L.pack6000(its, q)
    (d / (r["qid"] + ".json")).write_text(json.dumps({"question_id": r["qid"], "setting": q.setting, "messages": msgs,
                                                      "num_predict": npred, "messages_sha256": L.S10.sha256_msgs(msgs),
                                                      "evidence_chars": ev.chars}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(r["qid"], q.setting, ev.chars, flush=True)
