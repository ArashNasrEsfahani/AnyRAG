"""Convert datasets-v2 questions (question-v2, work/datasets_v2/SCHEMA.json) into the runner's question format, and
verify the converted file mechanically (the v2 equivalent of verify_multidoc.py for run files).

    .venv/Scripts/python.exe -m ragbench.tools.verify_v2run convert --src work/datasets_v2/questions/rpt_dev.jsonl \
        --out <runner.jsonl> [--frozen work/datasets_v2/FROZEN_questions.json] [--experiment v2-dev]
    .venv/Scripts/python.exe -m ragbench.tools.verify_v2run verify --questions <runner.jsonl> [--frozen ...]

Field mapping question-v2 -> runner (contracts.Question; a multi-document IDENTITY run: meta.experiment "multidoc"):
  id                         -> id (unchanged)
  domain + language          -> corpus: RPT-fa C, RPT-en D, BOOK K, LAW L
  question                   -> question_fa or question_en (the other is ""; the setting is <lang>-<lang>)
  expected_answer            -> answer_fa or answer_en (the other is the English gloss for Persian, else "")
  answerable                 -> answer_type "multi" (answerable) | "unanswerable"; type "multidoc" | "unanswerable"
  evidence[].unit_id         -> gold_units {lang: [unit ids, first-mention order]}
  doc_scope.primary "pool"   -> doc_scope "corpus" (the frozen pool masks it; oracle_docs stay evaluator-only in meta)
  pool_id                    -> pool_id (frozen manifest work/multidoc/pools/<pool_id>.json)
  answer_format              -> answer_format v2_<format> (contracts.v2_runner_format; "abstain" -> the format of the
                                intended task, so a method never learns answerability from the prompt)
  schema                     -> schema "question-v2-run"; meta.experiment "multidoc", meta.experiment_id
  everything else            -> meta.v2 (the complete source record, evaluator-only: methods receive a Query, never
                                the Question); meta.v2_source {file, file_sha256, content_hash}
Methods see: the question text in its language, the pool (as a scope mask) and the answer format's prompt. Nothing else.

verify checks (errors): Question.validate(); multidoc identity; runner format = v2_runner_format(source); corpus and
setting agree with domain/language; question text only in its language; doc_scope "corpus"; pool manifest loads, its
corpus matches, every evidence document is in the pool and the pool is not the gold documents; every gold unit exists;
every evidence quote equals the frozen document text at [start, end) and lies inside its unit; content_hash and source
file sha256 equal the freeze file (when given); unanswerable: no gold units. Flags: pool == gold docs, empty glosses.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ..contracts import V2_FORMATS, Question, v2_runner_format

CORPUS = {("RPT", "fa"): "C", ("RPT", "en"): "D", ("BOOK", "fa"): "K", ("BOOK", "en"): "K",
          ("LAW", "fa"): "L", ("LAW", "en"): "L"}
RUN_SCHEMA = "question-v2-run"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def convert_record(r, src_name, src_sha, experiment_id):
    lang = r["language"]
    other = "en" if lang == "fa" else "fa"
    units = []
    for e in r["evidence"]:
        if e["unit_id"] not in units:
            units.append(e["unit_id"])
    q = {"id": r["id"], "corpus": CORPUS[(r["domain"], lang)],
         "question_%s" % lang: r["question"], "question_%s" % other: "",
         "answer_%s" % lang: r["expected_answer"],
         "answer_%s" % other: (r.get("expected_answer_gloss_en") or "") if lang == "fa" else "",
         "answer_type": "multi" if r["answerable"] else "unanswerable",
         "gold_units": {lang: units} if r["answerable"] else {},
         "doc_scope": "corpus",
         "type": "multidoc" if r["answerable"] else "unanswerable",
         "schema": RUN_SCHEMA,
         "answer_format": v2_runner_format(r["answer_format"], r.get("intended_tier")),
         "pool_id": r["pool_id"],
         "meta": {"experiment": "multidoc", "experiment_id": experiment_id, "lang": lang,
                  "setting": "%s-%s" % (lang, lang), "tier": r["tier"], "intended_tier": r.get("intended_tier"),
                  "subtype": r.get("subtype"), "domain": r["domain"], "schema_answer_format": r["answer_format"],
                  "v2_source": {"file": src_name, "file_sha256": src_sha, "content_hash": r["content_hash"],
                                "revision": r["revision"]},
                  "v2": r}}
    return q


def load_frozen(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(questions, frozen=None, cfg=None):
    """-> (errors {id: [..]}, flags {id: [..]}, summary)."""
    from ..config import load_config
    from ..corpus.registry import load_set, set_name_for
    from ..pools import load_pool, resolve_pool_path
    cfg = cfg or load_config()
    sets, pools = {}, {}
    fz_q, fz_f = {}, {}
    if frozen:
        for fname, fv in frozen["files"].items():
            fz_f[Path(fname).name] = fv["sha256"]
            for qid, qv in fv["questions"].items():
                fz_q[qid] = qv
    errs, flags = {}, {}
    ids = Counter(q["id"] for q in questions)
    for d in questions:
        e, f = [], []
        qid = d.get("id")
        if ids[qid] > 1:
            e.append("duplicate id")
        q = Question.from_dict(d)
        bad = q.validate()
        if bad:
            e.append("validate: %s" % bad)
        if not q.is_multidoc():
            e.append("not an identity (multidoc) question")
        m = d.get("meta") or {}
        r = m.get("v2") or {}
        lang = r.get("language")
        if d.get("schema") != RUN_SCHEMA:
            e.append("schema %r" % d.get("schema"))
        if d.get("answer_format") not in V2_FORMATS:
            e.append("answer_format %r not in %s" % (d.get("answer_format"), V2_FORMATS))
        elif d["answer_format"] != v2_runner_format(r.get("answer_format"), r.get("intended_tier")):
            e.append("answer_format does not follow the source format %s" % r.get("answer_format"))
        if CORPUS.get((r.get("domain"), lang)) != d.get("corpus"):
            e.append("corpus %s does not match %s-%s" % (d.get("corpus"), r.get("domain"), lang))
        other = "en" if lang == "fa" else "fa"
        if not (d.get("question_%s" % lang) or "").strip() or (d.get("question_%s" % other) or "").strip():
            e.append("question text must be in %s only" % lang)
        if (d.get("question_%s" % lang) or "") != r.get("question"):
            e.append("question text differs from the source")
        if not q.supports("%s-%s" % (lang, lang)):
            e.append("question does not support setting %s-%s" % (lang, lang))
        if d.get("doc_scope") != "corpus":
            e.append("doc_scope must be 'corpus'")
        # pool --------------------------------------------------------------------------------------------------------
        pid = d.get("pool_id")
        if pid != r.get("pool_id"):
            e.append("pool_id differs from source")
        try:
            if pid not in pools:
                pools[pid] = load_pool(resolve_pool_path(pid, cfg))
            pool = pools[pid]
            if pool.corpus != d.get("corpus"):
                e.append("pool %s corpus %s != %s" % (pid, pool.corpus, d.get("corpus")))
            pdocs = set(pool.doc_ids())
            edocs = {x["doc_id"] for x in r.get("evidence") or []}
            if edocs - pdocs:
                e.append("evidence documents outside pool %s: %s" % (pid, sorted(edocs - pdocs)))
            if edocs and edocs == pdocs:
                e.append("pool equals the gold documents (gold-derived in effect)")
        except Exception as ex:  # noqa: BLE001
            e.append("pool %s: %s" % (pid, str(ex)[:160]))
        # evidence ----------------------------------------------------------------------------------------------------
        sname = set_name_for(d.get("corpus"), lang) if lang else None
        if sname and sname not in sets:
            sets[sname] = load_set(sname, cfg)
        cs = sets.get(sname)
        if r.get("answerable"):
            gold = (d.get("gold_units") or {}).get(lang) or []
            if sorted(set(gold)) != sorted({x["unit_id"] for x in r["evidence"]}):
                e.append("gold_units differ from the evidence unit ids")
            for x in r["evidence"]:
                try:
                    doc = cs.doc(x["doc_id"])
                    u = doc.unit(x["unit_id"])
                except Exception:  # noqa: BLE001
                    e.append("%s: unit %s not found" % (x["id"], x["unit_id"]))
                    continue
                s, t = x.get("start"), x.get("end")
                if s is None or t is None or doc.text[s:t] != x["quote"]:
                    e.append("%s: quote is not the document text at [%s, %s)" % (x["id"], s, t))
                elif not (u.span[0] <= s and t <= u.span[1]):
                    f.append("%s: quote extends outside unit span %s" % (x["id"], u.span))
                if x.get("text_sha256") and x["text_sha256"] != hashlib.sha256(doc.text.encode("utf-8")).hexdigest():
                    e.append("%s: text_sha256 differs from the frozen document text" % x["id"])
        else:
            if d.get("gold_units"):
                e.append("unanswerable with gold units")
            if d.get("answer_type") != "unanswerable":
                e.append("unanswerable needs answer_type unanswerable")
        # freeze ------------------------------------------------------------------------------------------------------
        if frozen is not None:
            fq = fz_q.get(qid)
            src = m.get("v2_source") or {}
            if fq is None:
                e.append("not in the freeze file")
            elif fq["content_hash"] != r.get("content_hash") or fq["revision"] != r.get("revision"):
                e.append("content_hash/revision differ from the freeze")
            if fz_f.get(src.get("file")) != src.get("file_sha256"):
                e.append("source file sha256 differs from the freeze")
        if lang == "fa" and not r.get("question_gloss_en"):
            f.append("no English gloss")
        if e:
            errs[qid] = e
        if f:
            flags[qid] = f
    summ = {"n": len(questions), "errors": len(errs), "flagged": len(flags),
            "by_pool": dict(Counter(q["pool_id"] for q in questions)),
            "by_format": dict(Counter(q["answer_format"] for q in questions)),
            "by_tier": dict(Counter(((q.get("meta") or {}).get("tier") or "AB") for q in questions))}
    return errs, flags, summ


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True, action="append")
    c.add_argument("--out", required=True)
    c.add_argument("--frozen", default=None)
    c.add_argument("--experiment", default="datasets-v2-dev")
    v = sub.add_parser("verify")
    v.add_argument("--questions", required=True)
    v.add_argument("--frozen", default=None)
    v.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        out = []
        for src in a.src:
            p = Path(src)
            sha = sha256_file(p)
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    out.append(convert_record(json.loads(l), p.name, sha, a.experiment))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out), encoding="utf-8")
        print("wrote %d questions -> %s" % (len(out), a.out))
        a = argparse.Namespace(questions=a.out, frozen=a.frozen, report=None)
    qs = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    errs, flags, summ = verify(qs, load_frozen(a.frozen))
    print(json.dumps(summ, ensure_ascii=False))
    for k, v_ in errs.items():
        print("ERROR", k, v_)
    for k, v_ in flags.items():
        print("flag", k, v_)
    if a.report:
        Path(a.report).write_text(json.dumps({"summary": summ, "errors": errs, "flags": flags}, ensure_ascii=False,
                                             indent=1), encoding="utf-8")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
