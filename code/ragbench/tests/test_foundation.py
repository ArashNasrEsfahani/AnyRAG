"""Foundation smoke test.

    .venv/Scripts/python.exe -m ragbench.tests.test_foundation            # offline checks only
    .venv/Scripts/python.exe -m ragbench.tests.test_foundation --ollama   # + 3 chat calls and 1 embed call
    add --scratch-extracted <dir> to exercise the extracted adapter on a partial extraction
    add --judge-key to query OpenRouter credit (GET /key, costs nothing; never prints the key)

Run from D:/KareOne/RAG Paper. Model-heavy: do not run --ollama while another model job runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench.config import load_config  # noqa: E402
from ragbench import normalize as N  # noqa: E402
from ragbench.corpus import load_set, swap_lang  # noqa: E402
from ragbench.store import build_plain_store, grep, read_lines, rg_binary, section_file  # noqa: E402
from ragbench.contracts import (Approach, AnswerResult, Question, RetrievedItem, Setting, RunLog,  # noqa: E402
                                pack_evidence, build_answer_messages)
from ragbench.llm import OllamaClient, Ledger, OpenRouterClient  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail else ""), flush=True)
    return cond


def test_config(cfg):
    check("config.models", cfg.models["chat"] and cfg.models["embed"], cfg.models["chat"] + " / " + cfg.models["embed"])
    import subprocess
    out = subprocess.run([rg_binary(cfg), "--version"], capture_output=True).stdout.decode().splitlines()[0]
    check("ripgrep.binary", out.startswith("ripgrep"), out)


def test_normalize():
    sys.path.insert(0, str(load_config().path("project_root") / "tools" / "extract"))
    try:
        import fa_text
    except Exception as e:  # noqa: BLE001
        return check("normalize.parity", False, "cannot import extractor fa_text: %s" % e)
    samples = ["كتاب عربي", "می\u200c\u200cخواهم", "نیم\u200cسال اول ۱۳۹۸", "اسـت", "٣٫٥٪", "\u200fسلام\u200e دنیا",
               "ﻻ ﺑﺮﺍﯼ", "Digikala’s user base", "a\u00adb", " \u200cابتدا", "۱5 میلیون", "word\u200cword",
               "\ufeffBOM", "کسب\u00acوکار"]
    A = load_set("A")
    for d in A.docs[:4]:
        samples += [u.text_raw for u in d.units[:300]]
    bad = [s for s in samples if N.normalize(s) != fa_text.normalize(s)]
    check("normalize.parity_with_extractor", not bad, "%d strings, %d mismatches" % (len(samples), len(bad)))
    check("normalize.digits_yeh_kaf", N.normalize("كي ۱۲۳ ٤٥") == "کی 123 45", N.normalize("كي ۱۲۳ ٤٥"))
    check("detect_lang", N.detect_lang("سهم هوش مصنوعی از سرمایه AI") == "fa" and N.detect_lang("AI share of capital") == "en")


def test_corpus_A():
    t0 = time.time()
    A = load_set("A")
    check("A.loaded", len(A) == 20, "%s in %.1fs; skipped=%s" % (A, time.time() - t0, A.skipped))
    aligned = True
    spans = True
    for d in A.docs:
        if d.lang != "fa":
            continue
        e = A.doc(swap_lang(d.id, "en"))
        if [u.local_id for u in d.units] != [u.local_id for u in e.units] or \
                [s.local_id for s in d.sections] != [s.local_id for s in e.sections]:
            aligned = False
        for doc in (d, e):
            for u in doc.units:
                if doc.text[u.span[0]:u.span[1]] != u.text:
                    spans = False
            for s in doc.sections:
                if doc.section_text(s.id) != doc.text[s.span[0]:s.span[1]]:
                    spans = False
    check("A.unit_ids_aligned_fa_en", aligned, "10 reports, identical section and unit local ids")
    check("A.spans_exact", spans)
    fa, en = A.doc("A.rpt-a.fa"), A.doc("A.rpt-a.en")
    u = fa.units[5]
    check("A.example_pair", True, "%s | %s || %s" % (u.id, u.text[:70], en.unit(swap_lang(u.id, "en")).text[:70]))
    unresolved = sum(len(d.meta.get("unresolved_figures", [])) for d in A.docs)
    check("A.figures_resolved", unresolved == 0, "unresolved placeholders: %d" % unresolved)
    return A


def test_extracted(scratch):
    cfg = load_config()
    out = {}
    for name in ("B", "C"):
        cs = load_set(name)
        out[name] = cs.stats()
        check("%s.real_extractions" % name, True, "%s; skipped %d (e.g. %s)" % (cs, len(cs.skipped), cs.skipped[:1]))
    if scratch:
        B = load_set("B", extracted_root=scratch, allow_partial=True)
        ok = len(B) >= 1
        detail = str(B)
        if ok:
            d = B.docs[0]
            ok = all(d.text[u.span[0]:u.span[1]] == u.text for u in d.units) and d.pair_id is not None
            detail += "; %s structure=%s sections=%d units=%d pair=%s" % (d.id, d.meta.get("structure"), len(d.sections),
                                                                       len(d.units), d.pair_id)
        check("B.scratch_partial_extraction_loads", ok, detail)
    return out


def test_store(A):
    t0 = time.time()
    en = load_set("A_en")
    root = build_plain_store(en)
    check("store.plain_A_en", (root / "manifest.json").is_file(), "%s in %.1fs" % (root, time.time() - t0))
    hits = grep(root, r"<english term>", doc_ids=["A.rpt-a.en"], tags=["body"], max_hits=20)
    ok = bool(hits) and all(h.unit_id for h in hits)
    if ok:
        h = hits[0]
        u = en.unit(h.unit_id)
        ok = h.text.lower().find("<english term>") >= 0 and h.text in u.text
        text, a, b, uids = read_lines(root, h.path, h.line - 2, h.line + 2)
        ok = ok and h.unit_id in uids and section_file(root, h.section_id) == h.path
    check("store.grep_maps_lines_to_units", ok, "%d hits; first %s" % (len(hits), hits[0].unit_id if hits else None))
    fa_root = build_plain_store(load_set("A_fa"))
    fh = grep(fa_root, "<persian term>", doc_ids=["A.rpt-a.fa"], tags=["body"], max_hits=5)
    check("store.grep_persian", bool(fh), "%d hits for the Persian term" % len(fh))
    return root


class SmokeApproach(Approach):
    """Minimal approach used only by this test: grep the plain store, pack, answer."""
    name = "SMOKE"

    def _build_index(self, corpus_set, index_dir):
        return build_plain_store(corpus_set)

    def _answer(self, query, index):
        hits = grep(index.payload, r"<english term>|<persian term>", doc_ids=query.doc_scope if query.doc_scope != "corpus" else None,
                    tags=["body"], max_hits=10)
        items, seen = [], set()
        for h in hits:
            if h.unit_id in seen:
                continue
            seen.add(h.unit_id)
            u = index.corpus.unit(h.unit_id)
            items.append(RetrievedItem(text=u.text, doc_id=u.doc_id, section_id=u.section_id, unit_ids=[u.id],
                                       kind="unit", span=list(u.span), rank=len(items)))
        ev = self.pack(items, index, budget=1200)
        return AnswerResult(answer_text=self.generate_answer(query, ev), evidence=ev, retrieved=items,
                            trace={"hits": len(hits)})


def smoke_question():
    return Question(id="smoke-sample-01", corpus="A",
                    question_fa="در سال ۲۰۲۴ چند تن محصول نمونه تولید شد؟",
                    question_en="How many tonnes of the sample product were made in 2024?",
                    answer_fa="264,926 تن", answer_en="264,926 t", answer_type="numeric",
                    gold_units={"fa": ["A.rpt-a.fa::front/front#lede"], "en": ["A.rpt-a.en::front/front#lede"]},
                    doc_scope=["A.rpt-a.en"], type="lookup", answer_value=264926, answer_unit="t")


def test_offline_contracts(A):
    q = smoke_question()
    check("question.validate", not q.validate(), q.validate())
    check("question.scope_swap", q.scope("fa") == ["A.rpt-a.fa"])
    en = load_set("A_en")
    items = [RetrievedItem(text=u.text, doc_id=u.doc_id, section_id=u.section_id, unit_ids=[u.id], span=list(u.span))
             for u in en.doc("A.rpt-a.en").units[:40]]
    ev = pack_evidence(items, en)
    cfg = load_config()
    check("evidence.budget", ev.chars <= cfg.evidence["max_chars"] and ev.n_included >= 1,
          "%d chars, %d/%d items, last cut=%s" % (ev.chars, ev.n_included, ev.n_offered, ev.last_item_cut))
    from ragbench.contracts import Query
    qq = Query(q.id, q.question_en, "en", "en", "en-en", "A", "A_en", ["A.rpt-a.en"])
    msgs = build_answer_messages(qq, ev)
    check("prompt.shape", msgs[0]["role"] == "system" and "NOT_FOUND" in msgs[0]["content"] and "Evidence:" in msgs[1]["content"])
    check("setting.parse", Setting.parse("fa→en").name == "fa-en" and Setting.parse("en-fa").cross_lingual)


def test_ollama(cfg):
    ledger = Ledger(path=str(cfg.ensure_dir("ledger_root") / "smoke_foundation.jsonl"))
    client = OllamaClient(cfg, ledger)
    h = client.health()
    check("ollama.health", h.get("reachable") and h.get("chat_present") and h.get("embed_present"),
          {k: h.get(k) for k in ("version", "chat_present", "embed_present", "chat_digest_match", "embed_digest_match")})
    # 1) chat through an Approach (shared prompt, evidence pack, ledger scope)
    ap = SmokeApproach(cfg, client)
    rec = ap.answer(smoke_question(), "en-en")
    check("ollama.chat_via_approach", rec.error is None and rec.llm_calls == 1 and rec.prompt_tokens > 0 and rec.completion_tokens > 0,
          "answer=%r calls=%d ptok=%d ctok=%d latency=%.0fms ev_chars=%d gold_in_evidence=%s err=%s" % (
              rec.answer_text[:80], rec.llm_calls, rec.prompt_tokens, rec.completion_tokens, rec.latency_ms,
              rec.evidence_chars, "A.rpt-a.en::front/front#lede" in rec.evidence_units, rec.error))
    RunLog(cfg.ensure_dir("runs_root") / "smoke_foundation.jsonl").append(rec)
    # 2) determinism + prompt-cache accounting: same short prompt twice
    msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
    with ledger.scope(phase="smoke", approach="smoke"):
        r1 = client.chat(msgs, purpose="determinism-1", num_predict=8)
        r2 = client.chat(msgs, purpose="determinism-2", num_predict=8)
    check("ollama.deterministic", r1.text == r2.text, "%r vs %r; prompt_tokens %d vs %d (a drop means prompt-cache reuse is not counted)"
          % (r1.text, r2.text, r1.prompt_tokens, r2.prompt_tokens))
    # 3) embeddings
    with ledger.scope(phase="smoke", approach="smoke"):
        er = client.embed(["سهم هوش مصنوعی از سرمایه", "AI share of capital"], purpose="embed-smoke")
    import numpy as np
    v = np.array(er.vectors)
    cos = float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1])))
    check("ollama.embed", v.shape == (2, 1024) and er.prompt_tokens > 0, "shape=%s tokens=%d cos(fa,en)=%.3f wall=%.0fms" % (
        v.shape, er.prompt_tokens, cos, er.wall_ms))
    lines = [json.loads(x) for x in open(ledger.path, encoding="utf-8") if x.strip()]
    mine = [x for x in lines if x.get("run_id") == ledger.run_id]
    ok = all(k in mine[0] for k in ("phase", "approach", "purpose", "prompt_tokens", "completion_tokens", "wall_ms"))
    check("ledger.records", ok and len(mine) == 4, "%d records this run: %s" % (
        len(mine), [(x["kind"], x["phase"], x["approach"], x["purpose"], x["prompt_tokens"], x["completion_tokens"], x["wall_ms"]) for x in mine]))
    # index build stats written by the base class
    st = ap.ensure_index("A_en").stats
    check("index.build_stats", st.get("complete") and "totals" in st, {"size_bytes": st.get("size_bytes"), "totals": st.get("totals")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", action="store_true")
    ap.add_argument("--scratch-extracted")
    ap.add_argument("--judge-key", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    test_config(cfg)
    test_normalize()
    A = test_corpus_A()
    test_extracted(args.scratch_extracted)
    test_store(A)
    test_offline_contracts(A)
    if args.ollama:
        test_ollama(cfg)
    if args.judge_key:
        oc = OpenRouterClient(cfg)
        try:
            info = oc.key_info()
            check("openrouter.key_info", info.get("limit_remaining") is not None or info.get("usage") is not None,
                  {"limit_remaining": info.get("limit_remaining"), "usage": info.get("usage"), "budget": oc.budget_state()})
        except Exception as e:  # noqa: BLE001
            check("openrouter.key_info", False, type(e).__name__)
    fails = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(fails), (": " + ", ".join(fails)) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
