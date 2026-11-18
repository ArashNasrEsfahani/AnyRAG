"""Is there a DEV-derivable discriminator inside the `quote_without_term` rejections?

Written 2026-09-21 by the decontamination agent, as the second half of the escape question. `v3_decontam.py
--pass v2guard` shows that v3's chosen test - "a plan term stands in the verifier's own definition" - has the same
precision on dev (0.38) as the rejections it leaves vetoed (0.38): on dev it separates nothing. Before v4 either
keeps or drops the escape, every other discriminator that a dev-only analyst could have proposed is measured on
the same rejections, from the same recorded runs, with no model call.

Population: every `guard:quote_without_term` rejection on a DEV record of an arm that ran the HARD veto
(`AGENTIC2` on the wide dev set, `AGENTIC3+verify_gate` on the narrow one) that would pass the verbatim guard with
the term test removed. Label: does the rejected value equal one of that DEV question's gold values (scale-aware)?

    .venv/Scripts/python.exe -m ragbench.tools.v3_decontam_alt
"""
import json
import re
import sys
from collections import Counter

from ..normalize import fold, normalize, word_tokens
from .v3_decontam import (REL, V2_ARMS, gold_values, load_questions, matches_gold, read_runs, term_rx_of)

STOP_FA = set("""از در به با که را این آن های ها هر یک دو سه چه چند چقدر کدام برای بر بی تا هم می شد شده است بود
بوده کرد کرده کند شود شوند دارد دارند داشت داشته طبق اساس گزارش بنا بنابر چند""".split())
STOP_EN = set("""the a an of in on for to and or is are was were be been by with at from how many much what which
who whom whose does do did report according says said total number percent share""".split())


def content_words(text, lang):
    stop = STOP_FA if lang == "fa" else STOP_EN
    return [w for w in word_tokens(text or "", folded=True) if len(w) > 2 and w not in stop]


def any_word_in(words, target_fold):
    return any(w in target_fold for w in words)


def features(it, cand, plan, rec, q, rx, all_cids_values):
    """Every discriminator a DEV-only analyst could have proposed, for one rejected item."""
    dfn = normalize(str(it.get("definition") or ""))
    quote = normalize(str(it.get("quote") or ""))
    qtext = rec.get("question_text") or ""
    f = {}
    f["def_term"] = bool(dfn) and any(r.search(dfn) for r in rx)                       # v3's own test
    qwords = content_words(q.get("question_%s" % rec["q_lang"]) or "", rec["q_lang"])
    dwords = content_words(q.get("question_%s" % rec["d_lang"]) or "", rec["d_lang"])
    f["def_qword"] = any_word_in(qwords + dwords, fold(dfn))
    f["quote_qword"] = any_word_in(qwords + dwords, fold(quote))
    f["def_metric"] = False
    metric = normalize(str(plan.get("metric") or ""))
    if metric:
        mw = content_words(metric, "fa" if re.search(r"[؀-ۿ]", metric) else "en")
        f["def_metric"] = any_word_in(mw, fold(dfn))
    feats = (cand or {}).get("features") or {}
    src = set(feats.get("src") or [])
    f["src_facts"] = bool(src & {"facts", "factdb"})
    f["kind_ok"] = bool((feats.get("db") or {}).get("kind") == "match")
    v = it.get("value_num")
    f["corroborated"] = v is not None and sum(
        1 for other in all_cids_values if other is not None and abs(other - v) <= REL * max(abs(v), 1e-12)) > 1
    return f


def main():
    from ..approaches.agentic import verify as V

    rows = []
    for label, d, runfile, qfile in V2_ARMS:
        qs = load_questions(qfile)
        for rec in read_runs(d, runfile):
            q = qs.get(rec["question_id"])
            if q is None:
                continue
            golds = gold_values(q)
            plan = rec["trace"]["stages"].get("plan") or {}
            rx = term_rx_of(plan)
            unit_kind = plan.get("unit") or "count"
            text_by_cid = {(it.get("meta") or {}).get("cid"): (it.get("text") or "")
                           for it in (rec.get("retrieved_units") or [])}
            cands = {c["cid"]: c for c in (rec["trace"]["stages"]["retrieve"].get("candidates") or [])}
            items = [it for b in ((rec["trace"]["stages"].get("verify") or {}).get("batches") or [])
                     for it in (b.get("items") or [])]
            all_vals = [x.get("value_num") for x in items]
            for it in items:
                if it.get("verdict") != "guard:quote_without_term":
                    continue
                snip = text_by_cid.get(it["cid"], "")
                if not snip:
                    continue
                ok, _r, info = V.guard(it, snip, unit_kind=unit_kind, term_rx=None)
                if not ok:
                    continue
                it = dict(it, value_num=info.get("value_num"))
                all_vals2 = all_vals + [info.get("value_num")]
                f = features(it, cands.get(it["cid"]), plan, rec, q, rx, all_vals2)
                rows.append({"arm": label, "qid": rec["question_id"], "setting": rec["setting"],
                             "cross": rec["q_lang"] != rec["d_lang"],
                             "gold": matches_gold(info.get("value_num"), golds), **f})

    n = len(rows)
    ng = sum(1 for r in rows if r["gold"])
    print("\n# Dev-derivable discriminators inside the quote_without_term rejections\n")
    print("  population: %d escape-eligible rejections over the dev hard-veto arms; %d carry a dev gold value"
          " (base rate %.2f)\n" % (n, ng, ng / max(1, n)))
    keys = ["def_term", "def_qword", "quote_qword", "def_metric", "src_facts", "kind_ok", "corroborated"]
    print("  %-14s %5s %5s %6s | %5s %5s %6s" % ("rule fires", "gold", "wrong", "prec", "gold", "wrong", "prec"))
    print("  %-14s %19s | %19s" % ("", "-- rule TRUE --", "-- rule FALSE --"))
    for k in keys:
        a = [r for r in rows if r[k]]
        b = [r for r in rows if not r[k]]
        ga, gb = sum(1 for r in a if r["gold"]), sum(1 for r in b if r["gold"])
        pa = ("%.2f" % (ga / len(a))) if a else "-"
        pb = ("%.2f" % (gb / len(b))) if b else "-"
        print("  %-14s %5d %5d %6s | %5d %5d %6s" % (k, ga, len(a) - ga, pa, gb, len(b) - gb, pb))
    print("\n  cross-lingual dev rejections only (%d of %d)\n"
          % (sum(1 for r in rows if r["cross"]), n))
    cr = [r for r in rows if r["cross"]]
    for k in keys:
        a = [r for r in cr if r[k]]
        ga = sum(1 for r in a if r["gold"])
        print("  %-14s fires %2d, gold %2d" % (k, len(a), ga))
    print("\n  every eligible rejection, for inspection\n")
    for r in rows:
        print("   %-6s %-22s %-6s gold=%-5s %s" % ("GOLD" if r["gold"] else "wrong", r["qid"], r["setting"],
              r["gold"], " ".join("%s=%d" % (k, int(r[k])) for k in keys)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
