"""S12 report: scores, paired comparisons with the frozen 1,000-char HYBRID_6000 / HR_6000, tier/language/genre
breakdowns, gold-bundle coverage (S7 text method + an offset method), chunk counts, GPU time, tokens. No model call.
Frozen cells: report_test.load/score. New cells: judge/ dirs via report_test.final_verdicts_by_answer (S2a's policy).
Statistics: revision_stats.py helpers extracted with ast and executed unchanged (as S2a/report.py and S7 do).
-> results.json, tables.md"""
import ast
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import packs as PK

TRD = str(PK.TR)
os.environ["ANYRAG_TEST_DIR"] = TRD
os.environ["ANYRAG_FREEZE_DIR"] = str(PK.FZ)
os.environ["ANYRAG_CODE_DIR"] = PK.FROZEN
sys.path.insert(0, TRD)
import report_test as RT  # noqa: E402

sys.path.insert(0, str(PK.HERE.parent / "S7_analysis"))
import s7lib as L7  # noqa: E402  (gold_reach: S7's coverage method, used unchanged)

P = RT.PRIMARY
REC, VER = RT.load()
QS, ANS, AB = RT.QS, RT.ANS, RT.AB
HERE = PK.HERE


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.is_file() else []


SRC = PK.ROOT / "paper" / "analysis" / "revision_stats.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
want = {"find", "sign_p", "diff_vectors", "sign_range", "boot_mean", "compare"}
mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], type_ignores=[])
NS = {"RT": RT, "itertools": __import__("itertools"), "random": __import__("random"), "defaultdict": defaultdict,
      "N_BOOT": 10000, "SEED": 0, "ANS": ANS, "parent": {}}
exec(compile(mod, str(SRC), "exec"), NS)
QROWS = {}
for line in open(PK.FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
constituents = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
for q in ANS:
    c = sorted(constituents[q])
    for o in c[1:]:
        NS["parent"][NS["find"](o)] = NS["find"](c[0])
NS["comp"] = {q: NS["find"](sorted(constituents[q])[0]) for q in ANS}
NS["block"] = {q: QROWS[q]["source_block"] for q in ANS}
compare = NS["compare"]

ANSW = {}
for r in jl(HERE / "answers.jsonl"):
    if not r.get("error"):
        ANSW[(r["arm"], r["qid"])] = r
import judge as JG  # noqa: E402

NV = {}
jroot = HERE / "judge"
if jroot.is_dir():
    for d in [jroot] + sorted(p for p in jroot.iterdir() if p.is_dir()):
        for k, v in RT.final_verdicts_by_answer(d).items():
            NV.setdefault(k, v)
from ragbench.tools import judge_multi as JM  # noqa: E402

FROZEN = ["HYBRID_6000", "HR_6000"]
NEW = ["HYBRID_C500_6000", "HR_C500_6000", "HYBRID_C2000_6000", "HR_C2000_6000"]
ARMS = ["HYBRID_C500_6000", "HYBRID_6000", "HYBRID_C2000_6000", "HR_C500_6000", "HR_6000", "HR_C2000_6000"]
SIZE = {"HYBRID_6000": 1000, "HR_6000": 1000, "HYBRID_C500_6000": 500, "HR_C500_6000": 500,
        "HYBRID_C2000_6000": 2000, "HR_C2000_6000": 2000}


def score(arm, q):
    if arm in RT.ARMS:
        lo, hi, _ = RT.score(REC, VER, P, arm, q)
        return lo, hi
    r = ANSW.get((arm, q))
    if r is None:
        return 0, 1
    if r["source"].startswith("frozen:"):
        lo, hi, _ = RT.score(REC, VER, P, r["source"].split(":", 1)[1], q)
        return lo, hi
    rid = JG.identity(arm, q, r["setting"])["rid"]
    v = NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]})))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def status(arm, q):
    if arm in RT.ARMS:
        return RT.score(REC, VER, P, arm, q)[2]
    r = ANSW.get((arm, q))
    if r is None:
        return "missing"
    if r["source"].startswith("frozen:"):
        return "frozen"
    rid = JG.identity(arm, q, r["setting"])["rid"]
    v = NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]})))
    return "unjudged" if v is None else v["status"]


def frozen_answer_calls(arm):
    out = defaultdict(list)
    for f in (PK.TR / "runs").glob("*/%s/ledger_*.jsonl" % arm):
        for c in jl(f):
            if c.get("kind") == "chat" and c.get("phase") == "query":
                out[c["question_id"]].append(c)
    return out


FCALLS = {a: frozen_answer_calls(a) for a in FROZEN}
MYL = defaultdict(list)
for c in jl(HERE / "ledger.jsonl"):
    if c.get("purpose") == "answer" and c.get("ok"):
        MYL[(c["approach"], c["question_id"])].append(c)


def tokens(arm, q):
    """-> (answer prompt tokens, completion tokens, query LLM calls) or None."""
    if arm in FROZEN or (ANSW.get((arm, q)) or {}).get("source", "").startswith("frozen:"):
        fa = arm if arm in FROZEN else ANSW[(arm, q)]["source"].split(":", 1)[1]
        cs = [c for c in FCALLS[fa].get(q, []) if c.get("ok") and c.get("purpose") == "answer"]
        if not cs:
            return None
        return cs[-1]["prompt_tokens"], cs[-1]["completion_tokens"], len(FCALLS[fa].get(q, [])) if arm in FROZEN else 1
    r = ANSW.get((arm, q))
    if r is None:
        return None
    src = r["source"].split(":", 1)[1] if r["source"].startswith("dup:") else arm
    cs = MYL.get((src, q))
    if not cs:
        return None
    return cs[-1]["prompt_tokens"], cs[-1]["completion_tokens"], 1


PIDX = {(r["arm"], r["qid"]): r for r in jl(HERE / "packs_index.jsonl")}
TRACE = {(r["arm"], r["qid"]): r for r in jl(HERE / "pack_traces.jsonl")}


def chars(arm, q):
    if arm in FROZEN:
        r = REC[P][arm].get(q)
        return r["evidence_chars"] if r else None
    r = PIDX.get((arm, q))
    return r["chars"] if r else None


def messages(arm, q):
    if arm in FROZEN:
        return L7.messages(arm, q)
    f = HERE / "packs" / arm / ("%s.json" % q)
    return json.loads(f.read_text(encoding="utf-8"))["messages"] if f.is_file() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def tot(arm, qs=ANS):
    return sum(score(arm, q)[0] for q in qs), sum(score(arm, q)[1] for q in qs)


GROUPS = (("tier", lambda q: QS[q]["tier"]), ("lang", lambda q: QS[q]["lang"]), ("genre", lambda q: QS[q]["domain"]))


# ---------------------------------------------------------------- coverage
def chunks_of(size, set_name):
    import retrieve as RV
    return RV.hx_of(set_name, size).chunks


SETQ = {r["qid"]: r["set"] for r in jl(HERE / "search_1000.jsonl")}


def offset_reach(q, spans):
    """spans: [(doc_id, a, b)] -> 'complete' | 'partial' | 'none' over the annotated bundles; an evidence item is
    present when >= 90% of its [start, end) characters lie inside the union of the given same-document spans."""
    v2 = QROWS[q]
    by = defaultdict(list)
    for d, a, b in spans:
        by[d].append((a, b))
    pres = {}
    for e in v2["evidence"]:
        a, b = int(e["start"]), int(e["end"])
        cov = [0] * max(0, b - a)
        for x, y in by.get(e["doc_id"], []):
            for i in range(max(a, x), min(b, y)):
                cov[i - a] = 1
        pres[e["id"]] = (sum(cov) / len(cov) if cov else 0.0) >= 0.9
    fr = [sum(pres[x] for x in bun) / len(bun) for bun in v2["evidence_bundles"]]
    best = max(fr) if fr else 0.0
    return "complete" if best == 1.0 else ("partial" if best > 0 else "none")


def coverage():
    out = {"text_pack": {}, "offset_pack": {}, "offset_top40": {}, "offset_top8": {}, "per_q": {}}
    for arm in ARMS:
        size = SIZE[arm]
        txt, offp, off40, off8 = {}, {}, {}, {}
        for q in ANS:
            m = messages(arm, q)
            txt[q] = L7.gold_reach(q, m)["cls"]
            tr = TRACE.get((arm, q))
            ch = chunks_of(size, SETQ[q])
            offp[q] = offset_reach(q, [(p["doc_id"], p["span"][0], p["span"][1]) for p in tr["packed"]])
            top = [ch[i] for i in tr["top40"]]
            off40[q] = offset_reach(q, [(c["doc_id"], c["span"][0], c["span"][1]) for c in top])
            off8[q] = offset_reach(q, [(c["doc_id"], c["span"][0], c["span"][1]) for c in top[:8]])
        for key, d in (("text_pack", txt), ("offset_pack", offp), ("offset_top40", off40), ("offset_top8", off8)):
            row = {"all": dict(Counter(d.values()))}
            for g, f in GROUPS:
                for v in sorted({f(q) for q in ANS}):
                    row["%s=%s" % (g, v)] = dict(Counter(d[q] for q in ANS if f(q) == v))
            out[key][arm] = row
        out["per_q"][arm] = {q: [txt[q], offp[q], off40[q]] for q in ANS}
        # agreement of the two pack measures
        out.setdefault("text_vs_offset_agree", {})[arm] = sum(txt[q] == offp[q] for q in ANS)
    return out


# ---------------------------------------------------------------- index facts
def index_facts():
    import retrieve as RV
    res = {"chunks": {}, "chunk_chars": {}, "gpu": {}}
    for size in (500, 1000, 2000):
        for s in ["C", "D_en", "BOOK_fa", "BOOK_en", "LAW_fa", "LAW_en"]:
            hx = RV.hx_of(s, size)
            m = json.loads((hx.index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
            res["chunks"]["%d|%s" % (size, s)] = m["n_chunks"]
            res["chunk_chars"]["%d|%s" % (size, s)] = {k: m[k] for k in ("chars_mean", "chars_p50", "chars_max",
                                                                         "n_window", "n_packed")}
    scope = defaultdict(set)
    for size in (500, 1000, 2000):
        for r in jl(HERE / ("search_%d.jsonl" % size)):
            scope["%d|%s" % (size, r["set"])].add(r["chunks_in_scope"])
    res["chunks_in_question_scope"] = {k: sorted(v) for k, v in scope.items()}
    for size in (500, 2000):
        b = [r for r in jl(HERE / ("index_%d.jsonl" % size)) if r["phase"] == "build"]
        last = {}
        for r in b:
            last[r["set"]] = r
        res["gpu"]["index_%d" % size] = {
            "per_set_wall_s": {s: r["wall_s"] for s, r in last.items()},
            "wall_s_total": round(sum(r["wall_s"] for r in last.values()), 1),
            "ollama_embed_ms_total": round(sum((r.get("embed_stats") or {}).get("wall_ms_now", 0) for r in b), 1),
            "embedded_texts": sum((r.get("embed_stats") or {}).get("embedded_now", 0) for r in b),
            "embed_tokens": sum((r.get("embed_stats") or {}).get("tokens_now", 0) for r in b),
            "probe": [r for r in jl(HERE / ("index_%d.jsonl" % size)) if r["phase"] == "probe"]}
    for size in (500, 2000):     # every Ollama embed call (crashed/killed partial runs included), from the ledger
        led = [c for c in jl(HERE / ("ollama_ledger_%d.jsonl" % size)) if c.get("kind") == "embed"]
        ok = [c for c in led if c.get("ok")]
        g = res["gpu"]["index_%d" % size]
        g["ledger_embed_calls_ok"] = len(ok)
        g["ledger_embed_calls_failed"] = len(led) - len(ok)
        g["ledger_texts_embedded"] = sum(c.get("n_inputs") or 0 for c in ok)
        g["ledger_embed_tokens"] = sum(c.get("prompt_tokens") or 0 for c in ok)
        g["ledger_ollama_wall_s"] = round(sum(c.get("wall_ms") or 0 for c in ok) / 1000.0, 1)
        g["ledger_first_last"] = [led[0]["ts"], led[-1]["ts"]] if led else None
        g["retries"] = [r for r in jl(HERE / ("index_%d.jsonl" % size)) if r["phase"] == "retry"]
    res["gpu"]["phases"] = jl(HERE / "gpu_phases.jsonl")
    rs = defaultdict(lambda: [0.0, 0, 0])
    for r in jl(HERE / "rerank_scores.jsonl"):
        x = rs[r["size"]]
        x[0] += r["ms"]
        x[1] += r["tokens_nonpad"]
        x[2] += r.get("tokens_padded") or 0
    res["gpu"]["rerank_by_size"] = {k: {"ms": round(v[0], 1), "tokens_nonpad": v[1], "tokens_padded": v[2]}
                                    for k, v in rs.items()}
    return res


def main():
    res = {"table": [], "groups": {}, "compare": {}}
    for a in ARMS:
        t = [tokens(a, q) for q in QS]
        res["table"].append({
            "arm": a, "size": SIZE[a], "acc90": tot(a), "unresolved": sum(1 for q in ANS if score(a, q)[0] != score(a, q)[1]),
            "status": dict(Counter(status(a, q) for q in QS)), "abstain_correct6": tot(a, AB),
            "chars_mean": mean([chars(a, q) for q in QS]),
            "items_mean": mean([len(TRACE[(a, q)]["packed"]) for q in QS if (a, q) in TRACE]),
            "answer_prompt_tokens_mean": mean([x[0] if x else None for x in t]),
            "answer_prompt_tokens_en": mean([x[0] if x else None for x, q in zip(t, QS) if QS[q]["lang"] == "en"]),
            "answer_prompt_tokens_fa": mean([x[0] if x else None for x, q in zip(t, QS) if QS[q]["lang"] == "fa"]),
            "completion_tokens_mean": mean([x[1] if x else None for x in t]),
            "llm_calls_mean": mean([x[2] if x else None for x in t]), "tokens_unknown": sum(x is None for x in t)})
        g = {}
        for key, f in GROUPS:
            gg = defaultdict(list)
            for q in ANS:
                gg[f(q)].append(q)
            g[key] = {k: [*tot(a, v), len(v)] for k, v in sorted(gg.items())}
        res["groups"][a] = g
    pairs = []
    for s in ("C500", "C2000"):
        for p in ("HYBRID", "HR"):
            pairs += [("%s_%s_6000" % (p, s), "HYBRID_6000"), ("%s_%s_6000" % (p, s), "HR_6000")]
        pairs.append(("HR_%s_6000" % s, "HYBRID_%s_6000" % s))
    pairs += [("HYBRID_C2000_6000", "HYBRID_C500_6000"), ("HR_C2000_6000", "HR_C500_6000"), ("HR_6000", "HYBRID_6000")]
    for a, b in pairs:
        k = sum((score(a, q)[0] != score(a, q)[1]) + (score(b, q)[0] != score(b, q)[1]) for q in ANS)
        if k > 16:
            res["compare"]["%s vs %s" % (a, b)] = {"skipped": "unresolved cells %d > 16" % k}
            continue
        c = compare(lambda q, a=a: score(a, q), lambda q, b=b: score(b, q))
        sub = {}
        for key, f in GROUPS:
            for v in sorted({f(q) for q in ANS}):
                qq = [q for q in ANS if f(q) == v]
                w = sum(1 for q in qq if score(a, q)[0] == score(a, q)[1] == 1 and score(b, q)[0] == score(b, q)[1] == 0)
                l_ = sum(1 for q in qq if score(a, q)[0] == score(a, q)[1] == 0 and score(b, q)[0] == score(b, q)[1] == 1)
                sub["%s=%s" % (key, v)] = [w, l_]
        c["wins_losses_resolved_by_group"] = sub
        res["compare"]["%s vs %s" % (a, b)] = c
    res["coverage"] = coverage()
    res["index"] = index_facts()
    res["same_as_frozen_packs"] = {a: sum(1 for (x, q), r in PIDX.items() if x == a and r["same_as_frozen"]) for a in NEW}
    res["last_item_cut"] = {a: sum(1 for (x, q), r in PIDX.items() if x == a and r["last_item_cut"]) for a in NEW}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(res)


def write_md(res):
    md = ["# S12 tables (generated by report.py from results.json)", "",
          "## Accuracy (/90, bounds over unresolved judge cells), abstention (/6), pack size, reader tokens", "",
          "| arm | chunk | correct /90 | unresolved | abstain /6 | pack chars | items | reader prompt tok (all/en/fa) | completion tok | LLM calls/q |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["table"]:
        lo, hi = r["acc90"]
        md.append("| %s | %d | %s | %d | %s | %s | %s | %s / %s / %s | %s | %s |" % (
            r["arm"], r["size"], ("%d" % lo) if lo == hi else "%d-%d" % (lo, hi), r["unresolved"],
            "%d-%d" % tuple(r["abstain_correct6"]) if r["abstain_correct6"][0] != r["abstain_correct6"][1] else r["abstain_correct6"][0],
            r["chars_mean"], r["items_mean"], r["answer_prompt_tokens_mean"], r["answer_prompt_tokens_en"],
            r["answer_prompt_tokens_fa"], r["completion_tokens_mean"], r["llm_calls_mean"]))
    for key in ("tier", "lang", "genre"):
        ks = list(res["groups"][ARMS[0]][key])
        md += ["", "## By %s (correct lo-hi / n)" % key, "", "| arm | " + " | ".join(ks) + " |",
               "|---|" + "---|" * len(ks)]
        for a in ARMS:
            g = res["groups"][a][key]
            md.append("| %s | %s |" % (a, " | ".join(("%d" % g[k][0] if g[k][0] == g[k][1] else "%d-%d" % (g[k][0], g[k][1]))
                                                     + "/%d" % g[k][2] for k in ks)))
    md += ["", "## Paired comparisons (A vs B; net = wins - losses, bounds; sign p range; 95% bootstrap CI in pp)", "",
           "| A vs B | net | wins/losses (resolved) | sign p | CI questions | CI components |", "|---|---|---|---|---|---|"]
    for k, v in res["compare"].items():
        if "skipped" in v:
            md.append("| %s | %s | | | | |" % (k, v["skipped"]))
            continue
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            k, v["net_bounds"], v["better_worse_resolved"], [round(x, 4) for x in v["sign_p_range"]],
            [round(x, 1) for x in v["ci_pp_question"]], [round(x, 1) for x in v["ci_pp_component"]]))
    first = [v for v in res["compare"].values() if "skipped" not in v]
    ks = list(first[0]["wins_losses_resolved_by_group"]) if first else []
    md += ["", "### Resolved wins/losses by group", "", "| A vs B | " + " | ".join(ks) + " |"]
    md.append("|---|" + "---|" * len(ks))
    for k, v in res["compare"].items():
        if "skipped" not in v:
            md.append("| %s | %s |" % (k, " | ".join("+%d/-%d" % tuple(v["wins_losses_resolved_by_group"][x]) for x in ks)))
    cov = res["coverage"]
    for key, title in (("text_pack", "S7 text method: gold bundle in the reader's pack"),
                       ("offset_pack", "offset method: gold bundle inside the packed chunk spans"),
                       ("offset_top8", "offset method: gold bundle inside the top-8 ranked chunks"),
                       ("offset_top40", "offset method: gold bundle inside the top-40 ranked chunks")):
        ks = [k for k in cov[key][ARMS[0]] if k != "all"]
        md += ["", "## Coverage, %s (complete / partial / none of 90)" % title, "",
               "| arm | all | " + " | ".join(ks) + " |", "|---|---|" + "---|" * len(ks)]
        for a in ARMS:
            row = cov[key][a]
            f = lambda d: "%d/%d/%d" % (d.get("complete", 0), d.get("partial", 0), d.get("none", 0))  # noqa: E731
            md.append("| %s | %s | %s |" % (a, f(row["all"]), " | ".join(f(row[k]) for k in ks)))
    md += ["", "Text vs offset pack measure agree on: %s" % cov["text_vs_offset_agree"]]
    ix = res["index"]
    md += ["", "## Chunks per pool (= corpus set; every test question's scope is its whole pool)", "",
           "| pool (set) | 500 | 1000 | 2000 | mean chars 500/1000/2000 |", "|---|---|---|---|---|"]
    pools = {"C": "RPT-fa", "D_en": "RPT-en", "BOOK_fa": "BOOK-fa", "BOOK_en": "BOOK-en", "LAW_fa": "LAW-fa", "LAW_en": "LAW-en"}
    tot_ = Counter()
    for s, p in pools.items():
        n = [ix["chunks"]["%d|%s" % (z, s)] for z in (500, 1000, 2000)]
        for z, x in zip((500, 1000, 2000), n):
            tot_[z] += x
        md.append("| %s (%s) | %d | %d | %d | %s |" % (p, s, *n, " / ".join(str(ix["chunk_chars"]["%d|%s" % (z, s)]["chars_mean"]) for z in (500, 1000, 2000))))
    md.append("| total | %d | %d | %d | |" % (tot_[500], tot_[1000], tot_[2000]))
    md += ["", "## GPU", "", "```", json.dumps(ix["gpu"], indent=1), "```"]
    md += ["", "Packs identical to a frozen pack: %s; packs whose last item was cut: %s" % (
        res["same_as_frozen_packs"], res["last_item_cut"])]
    (HERE / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
