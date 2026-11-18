"""The versioned multi-document question schema (multidoc-q-v1) and its mechanical verifier (2026-09-24, P0 list of the
multi-document design, work/analysis/MULTIDOC_ASTRA_session.log Q2/Q3/Q6). verify_fresh.py (one scoped report per
question) stays as it was for the finished study.

    .venv/Scripts/python.exe -m ragbench.tools.verify_multidoc --questions <file.jsonl> [--pools work/multidoc/pools]
        [--no-corpus] [--out work/multidoc/verify]

A multidoc-q-v1 line is an ordinary question line (contracts.Question: id, corpus, question_fa/en, answer_fa/en,
answer_type, gold_units, doc_scope, type, meta) plus:
  schema                "multidoc-q-v1"
  task_family           single_fact | routing | series | cross_year | comparison | cross_company | aggregation | list |
                        within_report_list | unanswerable            (contracts.TASK_FAMILY_FORMATS)
  answer_format         optional; single | series | comparison | list; must agree with task_family when both are given
  doc_scope             "corpus": the pool is the scope (never a gold-derived document list)
  pool_id               str or [str]: frozen pool manifest id(s) (ragbench/pools.py) the question runs in
  source_companies      [company keys] the answer draws on
  source_block          the source-company block (bootstrap and allocation unit; paraphrases stay in one block)
  information_need_id   the need a question expresses; paraphrases share it (and declare paraphrase_of)
  paraphrase_of         optional id of the question this one rewords
  content_units         [{claim_id, company, period, metric, definition, value, unit, role (operand | derived | item |
                        fact), source_spans: [{doc_id, unit_id, start, end, quote}]}]   (required content)
  accepted_evidence_sets [[doc ids]]: every alternative set of documents that fully supports the answer
  accepted_alternatives {claim_id: [accepted alternative wordings / values]}
  definitions           {metric: definition}   calculation_rules: text  (the judge's reference packet uses both)
  multidoc_necessity    {declared: true, indispensable_docs: [doc ids], single_doc_complete: false, audit: text}
                        (required for multi-document families)
  meta                  experiment "multidoc", experiment_id, lang, absent_search (unanswerable), entity_names

Checks - errors drop a question; flags are reported for the reviewer:
  schema       required fields, Question.validate(), known family/format, doc_scope "corpus", unique ids and texts
  pool         pool ids given; each manifest loads (frozen, not gold-derived); every source document is in the pool;
               a pool whose documents are exactly the question's gold documents is refused (gold-derived in effect)
  units        claim ids unique; company, period, metric, value and >= 1 source span per unit; span documents and
               unit ids exist and each quote equals the document text at [start, end) (with the corpus); span unit ids
               are in gold_units
  necessity    multi-document families: declared, >= 2 indispensable documents each supplying a unit, not answerable
               from one document (single_doc_complete false), every accepted evidence set has >= 2 documents, units
               from >= 2 documents
  unanswerable no content units or gold units, meta.absent_search non-empty
  needs        one information need in one source block; different needs with near-identical wording (error when
               identical after normalisation, flag above NEAR_DUP Jaccard)
  flags        copied_span (COPY_NGRAM consecutive words of a source quote in the question), answer_leakage (a unit
               value of 3+ digits in the question), lexical_overlap (question/quote word overlap outside
               [LOW_OVERLAP, HIGH_OVERLAP]), undeclared_paraphrase (a shared need without paraphrase_of)
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..contracts import MULTIDOC_FORMATS, MULTIDOC_SCHEMA, TASK_FAMILY_FORMATS, Question

MULTI_FAMILIES = {"series", "cross_year", "comparison", "cross_company", "aggregation", "list"}
REQUIRED = ("id", "corpus", "type", "answer_type", "task_family", "pool_id", "source_companies", "source_block",
            "information_need_id", "content_units", "accepted_evidence_sets")
UNIT_FIELDS = ("claim_id", "company", "period", "metric", "value")
COPY_NGRAM = 8
NEAR_DUP = 0.9
LOW_OVERLAP, HIGH_OVERLAP = 0.02, 0.6
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def words(text):
    return re.findall(r"[^\W_]+", (text or "").translate(_DIGITS).lower())


def question_texts(q):
    return [t for t in (q.get("question_fa"), q.get("question_en")) if (t or "").strip()]


def unit_docs(q):
    """Documents that supply the question's content units, in first-mention order."""
    out = []
    for u in q.get("content_units") or []:
        for sp in u.get("source_spans") or []:
            d = sp.get("doc_id") or (sp.get("unit_id") or "").split("::", 1)[0]
            if d and d not in out:
                out.append(d)
    return out


def gold_doc_sets(q):
    """The accepted evidence sets (each a set of doc ids that fully supports the answer); the documents of the content
    units when none are declared. Routing coverage is measured against these (outside the approaches)."""
    sets = [list(s) for s in (q.get("accepted_evidence_sets") or []) if s]
    return sets or ([unit_docs(q)] if unit_docs(q) else [])


def _norm_text(t):
    return " ".join(words(t))


def _ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def check_question(q, corpus=None, pools=None):
    """-> (errors, flags) for one question dict. corpus: {doc_id: Document} or an object with has_doc/doc; pools:
    {pool_id: Pool or exception}."""
    err, flag = [], []
    for k in REQUIRED:
        if q.get(k) in (None, "", []) and not (k in ("content_units", "accepted_evidence_sets", "source_companies")
                                               and q.get("task_family") == "unanswerable"):
            err.append("schema: %s missing" % k)
    if q.get("schema") != MULTIDOC_SCHEMA:
        err.append("schema: schema is %r, expected %s" % (q.get("schema"), MULTIDOC_SCHEMA))
    fam = q.get("task_family")
    if fam not in TASK_FAMILY_FORMATS:
        err.append("schema: unknown task_family %r" % fam)
    fmt = q.get("answer_format")
    if fmt is not None and fmt not in MULTIDOC_FORMATS:
        err.append("schema: answer_format %r is not a multi-document format %s" % (fmt, MULTIDOC_FORMATS))
    if fmt and fam in TASK_FAMILY_FORMATS and TASK_FAMILY_FORMATS[fam] != fmt:
        err.append("schema: answer_format %s disagrees with task_family %s (%s)" % (fmt, fam, TASK_FAMILY_FORMATS[fam]))
    if q.get("doc_scope") not in ("corpus", None):
        err.append("schema: doc_scope must be 'corpus' (the frozen pool is the scope), got a document list")
    try:
        bad = Question.from_dict(q).validate()
        if bad:
            err.append("schema: %s" % bad)
    except Exception as e:  # noqa: BLE001
        err.append("schema: %s" % str(e)[:120])
    if not question_texts(q):
        err.append("schema: no question text")
    unans = fam == "unanswerable" or q.get("answer_type") == "unanswerable"
    units = q.get("content_units") or []
    gold = q.get("gold_units") or {}
    gold_all = {u for l in gold.values() for u in (l or [])}
    # pool --------------------------------------------------------------------------------------------------------
    pids = [q["pool_id"]] if isinstance(q.get("pool_id"), str) else list(q.get("pool_id") or [])
    docs = unit_docs(q)
    for pid in pids:
        p = (pools or {}).get(pid) if pools is not None else None
        if pools is None:
            continue
        if p is None:
            err.append("pool: manifest %s not found" % pid)
            continue
        if isinstance(p, Exception):
            err.append("pool: %s does not load: %s" % (pid, str(p)[:160]))
            continue
        members = set(p.doc_ids())
        outside = [d for d in docs if d not in members]
        if outside and not unans:
            err.append("pool: %s lacks source document(s) %s" % (pid, outside[:5]))
        if docs and members == set(docs):
            err.append("pool: %s holds exactly the question's gold documents (a gold-derived pool)" % pid)
    # units -------------------------------------------------------------------------------------------------------
    if unans:
        if units or gold_all:
            err.append("unanswerable: has content units or gold units")
        if not (q.get("meta") or {}).get("absent_search"):
            err.append("unanswerable: meta.absent_search missing")
    ids = Counter(u.get("claim_id") for u in units)
    dup = [c for c, n in ids.items() if n > 1]
    if dup:
        err.append("units: duplicate claim ids %s" % dup)
    for u in units:
        miss = [k for k in UNIT_FIELDS if u.get(k) in (None, "")]
        if miss:
            err.append("units: %s lacks %s" % (u.get("claim_id"), miss))
        spans = u.get("source_spans") or []
        if not spans:
            err.append("units: %s has no source span" % u.get("claim_id"))
        for sp in spans:
            d = sp.get("doc_id") or (sp.get("unit_id") or "").split("::", 1)[0]
            if sp.get("unit_id") and gold_all and sp["unit_id"] not in gold_all:
                err.append("units: span unit %s of %s is not in gold_units" % (sp["unit_id"], u.get("claim_id")))
            if corpus is None:
                continue
            has = corpus.has_doc(d) if hasattr(corpus, "has_doc") else d in corpus
            if not has:
                err.append("units: document %s of %s does not exist" % (d, u.get("claim_id")))
                continue
            doc = corpus.doc(d) if hasattr(corpus, "doc") else corpus[d]
            if sp.get("unit_id") and not doc.has_unit(sp["unit_id"]):
                err.append("units: unit %s does not exist" % sp["unit_id"])
            if sp.get("start") is not None and sp.get("end") is not None and sp.get("quote") is not None:
                if doc.text[int(sp["start"]):int(sp["end"])] != sp["quote"]:
                    err.append("units: quote of %s differs from %s[%s:%s]" % (u.get("claim_id"), d, sp["start"], sp["end"]))
    # necessity ---------------------------------------------------------------------------------------------------
    if fam in MULTI_FAMILIES:
        nec = q.get("multidoc_necessity") or {}
        ind = list(nec.get("indispensable_docs") or [])
        if not nec.get("declared"):
            err.append("necessity: multidoc_necessity.declared is not true")
        if len(set(ind)) < 2:
            err.append("necessity: fewer than 2 indispensable documents")
        if nec.get("single_doc_complete") is not False:
            err.append("necessity: single_doc_complete must be false (no one document supports the whole answer)")
        not_used = [d for d in ind if d not in docs]
        if not_used:
            err.append("necessity: indispensable document(s) %s supply no content unit" % not_used)
        if len(set(docs)) < 2:
            err.append("necessity: content units come from %d document(s)" % len(set(docs)))
        for s in q.get("accepted_evidence_sets") or []:
            if len(set(s)) < 2:
                err.append("necessity: an accepted evidence set has one document %s" % list(s))
    # flags -------------------------------------------------------------------------------------------------------
    quotes = [sp.get("quote") or "" for u in units for sp in (u.get("source_spans") or [])]
    for t in question_texts(q):
        qw = words(t)
        qg = _ngrams(qw, COPY_NGRAM)
        if qg and any(qg & _ngrams(words(x), COPY_NGRAM) for x in quotes):
            flag.append("copied_span: %d+ consecutive words of a source quote in the question" % COPY_NGRAM)
        nt = "".join(ch for ch in t.translate(_DIGITS) if ch.isdigit() or ch in ".,/")
        for u in units:
            v = re.sub(r"[^\d]", "", str(u.get("value") or "").translate(_DIGITS))
            if len(v) >= 3 and u.get("role") != "given" and v in re.sub(r"[^\d]", "", nt):
                flag.append("answer_leakage: value of %s appears in the question" % u.get("claim_id"))
        if quotes:
            qs_, ws_ = set(qw), set(w for x in quotes for w in words(x))
            ov = len(qs_ & ws_) / max(1, len(qs_))
            if ov < LOW_OVERLAP or ov > HIGH_OVERLAP:
                flag.append("lexical_overlap: %.2f outside [%.2f, %.2f]" % (ov, LOW_OVERLAP, HIGH_OVERLAP))
    return err, sorted(set(flag))


def check_set(questions, corpus=None, pools=None):
    """-> rows [{id, ok, errors, flags}] for a list of question dicts, with the cross-question checks (unique ids and
    texts, one need per source block, near-duplicate needs, undeclared paraphrases)."""
    rows = []
    id_n = Counter(q.get("id") for q in questions)
    text_n = Counter(_norm_text(" | ".join(question_texts(q))) for q in questions)
    need_blocks = defaultdict(set)
    need_members = defaultdict(list)
    for q in questions:
        need_blocks[q.get("information_need_id")].add(q.get("source_block"))
        need_members[q.get("information_need_id")].append(q)
    for q in questions:
        e, f = check_question(q, corpus, pools)
        if id_n[q.get("id")] > 1:
            e.append("schema: duplicate id")
        if text_n[_norm_text(" | ".join(question_texts(q)))] > 1:
            e.append("needs: duplicate question text")
        nid = q.get("information_need_id")
        if nid and len(need_blocks[nid] - {None}) > 1:
            e.append("needs: information need %s spans source blocks %s (paraphrases stay in one block)" % (
                nid, sorted(need_blocks[nid] - {None})))
        if nid and len(need_members[nid]) > 1 and not q.get("paraphrase_of") and \
                not any(o.get("paraphrase_of") == q.get("id") for o in need_members[nid]):
            f.append("undeclared_paraphrase: shares need %s without paraphrase_of" % nid)
        rows.append({"id": q.get("id"), "errors": e, "flags": f})
    # near-duplicate wording across different needs
    toks = [(q.get("id"), q.get("information_need_id"), set(words(" ".join(question_texts(q))))) for q in questions]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            if a[1] == b[1] or not a[2] or not b[2]:
                continue
            jac = len(a[2] & b[2]) / len(a[2] | b[2])
            if jac >= NEAR_DUP:
                for r in rows:
                    if r["id"] in (a[0], b[0]):
                        r["flags"].append("near_duplicate_need: %s ~ %s (Jaccard %.2f)" % (a[0], b[0], jac))
    for r in rows:
        r["flags"] = sorted(set(r["flags"]))
        r["ok"] = not r["errors"]
    return rows


def load_pools(pool_dir, ids):
    from ..pools import PoolError, load_pool
    out = {}
    for pid in ids:
        p = Path(pool_dir) / ("%s.json" % pid)
        if not p.is_file():
            continue
        try:
            out[pid] = load_pool(p)
        except (PoolError, OSError, ValueError) as e:
            out[pid] = e
    return out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    from ..config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pools", default=None, help="pool manifest dir (default work/multidoc/pools)")
    ap.add_argument("--no-corpus", action="store_true", help="skip the span checks against the corpus")
    ap.add_argument("--out", default=None, help="report dir (default work/multidoc/verify)")
    a = ap.parse_args(argv)
    cfg = load_config()
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8-sig") if l.strip()]
    pool_dir = Path(a.pools) if a.pools else Path(cfg.project_root) / "work" / "multidoc" / "pools"
    pids = sorted({p for q in qs for p in ([q["pool_id"]] if isinstance(q.get("pool_id"), str) else q.get("pool_id") or [])})
    pools = load_pools(pool_dir, pids)
    corpus = None
    if not a.no_corpus:
        from ..corpus.registry import load_set, set_name_for
        docs = {}
        for c in sorted({q.get("corpus") for q in qs if q.get("corpus")}):
            langs = ("fa",) if c == "C" else ("en",) if c == "D" else ("fa", "en")
            for l in langs:
                for d in load_set(set_name_for(c, l), cfg):
                    docs[d.id] = d
        corpus = docs
    rows = check_set(qs, corpus, pools)
    n_ok = sum(r["ok"] for r in rows)
    L = ["# Multi-document question verification (%s)" % MULTIDOC_SCHEMA, "", "Passing %d of %d questions; %d flagged." % (
        n_ok, len(rows), sum(1 for r in rows if r["flags"])), "", "## Errors", ""]
    L += ["- %s: %s" % (r["id"], "; ".join(r["errors"])) for r in rows if r["errors"]] or ["(none)"]
    L += ["", "## Flags (for the reviewer)", ""]
    L += ["- %s: %s" % (r["id"], "; ".join(r["flags"])) for r in rows if r["flags"]] or ["(none)"]
    out = Path(a.out) if a.out else Path(cfg.project_root) / "work" / "multidoc" / "verify"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.questions).stem
    (out / ("VERIFY_%s.md" % stem)).write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / ("verify_%s.json" % stem)).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L[:60]))
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
