"""Offline tests for LEX3 (no model, no network: scripted chat, hashed-token embeddings, a synthetic ten-report
English corpus; knowledge base, hybrid index and embedding cache in a throwaway dir under work/upgrades/lex3/).

    .venv/Scripts/python.exe -m ragbench.approaches.lex3.test_lex3

  prompts     interpret / pick validation: strict schema, problems reported, flat and garbage replies
  vocab       phrase-ID provenance: every phrase is an exact string of its source report, IDs unique, spread
  lexical     morphology is a lower-weight ADDITIONAL match (literal kept, no double credit, no derivational strip);
              name demotion only for the issuer inside its own report
  fusion      one contribution per channel per region; independent of channel order
  routing     shortlist <= 8, explicit entity/period matches kept, the escape channel survives (and no_escape drops it)
  session     3 calls; phrase provenance in the terms; requirement reservation; exact budget; empty-plan injection
              answers from the hybrid budget; call-2 failure keeps keyword + question channels; no shortlist skips
              call 2; the call cap; no document embedding at query time; variants card hook; reserved items marked
"""
import json
import re
import shutil
import sys
from pathlib import Path

from ...config import load_config
from ...contracts import Query, pack_evidence
from ...corpus.model import DocBuilder
from ...corpus.registry import CorpusSet
from ...llm import Ledger, ModelError
from ...normalize import fold
from ...retrieval import hybrid as H
from ...tests.test_baselines import FakeOllama
from ..lex import search as S
from . import core as C
from . import prompts as P
from .approach import Lex3Agent, _Session3, parse_flags
from .defaults import DEFAULTS

FAILS = []
N = [0]
DEFAULTS_EVIDENCE_CAP = __import__("ragbench.evidence", fromlist=["DEFAULTS"]).DEFAULTS["max_blocks"]
TMP = Path(r"D:\KareOne\RAG Paper\work\upgrades\lex3\test_tmp")


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label + ((" - " + str(detail)[:500]) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(label)
    return cond


# ---------------------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------------------

def filler(topic, n):
    return " ".join("Sentence %d on %s covers routine group matters in some detail for the reader." % (i, topic)
                    for i in range(n))


def company_doc(key, name, year, staff, revenue, mills, style="staff"):
    b = DocBuilder("T.%s_annual-report_%d.en" % (key, year), "T", "en", "%s annual report %d" % (name, year),
                   source="synthetic")
    b.add_section("intro", "Introduction")
    b.add_unit("intro", "intro#p0", "paragraph", "%s presents its results for %d. " % (name, year) + filler("the year", 4))
    b.add_section("people", "Our people")
    word = "staff" if style == "staff" else "employees"
    b.add_unit("people", "people#p0", "paragraph",
               filler("culture", 3) + " At year end the group employed %s %s across its plants. " % (staff, word)
               + filler("training", 3))
    b.add_unit("people", "people#cap", "caption", "Table 1: Headcount by plant")
    b.add_unit("people", "t1#table.0", "table", "Plant | %s\n%s | %s\n%s Mining Division | 310" % (year, mills, staff, name.split()[0]))
    b.add_section("fin", "Financial results")
    b.add_unit("fin", "fin#cap", "caption", "Table 2: Revenue by segment")
    b.add_unit("fin", "t2#table.0", "table", "Segment | Revenue %d\nSteel | %s\nMining | 50" % (year, revenue))
    b.add_unit("fin", "fin#p0", "paragraph", "%s revenue reached %s million in %d, while production increased. "
               % (name, revenue, year) + filler("results", 3))
    b.add_section("risk", "Risks")
    b.add_unit("risk", "risk#p0", "paragraph", filler("risks", 5))
    d = b.build()
    d.meta["fingerprint"] = "test-" + d.id
    return d


def other_doc(key, name, topic):
    b = DocBuilder("T.%s_annual-report_2022.en" % key, "T", "en", "%s annual report 2022" % name, source="synthetic")
    b.add_section("s1", "Overview of %s" % topic)
    b.add_unit("s1", "s1#p0", "paragraph", "%s focuses on %s. " % (name, topic) + filler(topic, 6))
    b.add_section("s2", "Outlook")
    b.add_unit("s2", "s2#p0", "paragraph", filler("the outlook for " + topic, 6))
    d = b.build()
    d.meta["fingerprint"] = "test-" + d.id
    return d


def corpus():
    docs = [company_doc("alphacorp", "Alpha Corp", 2021, "4,010", "1,410", "Alpha Steel Mills"),
            company_doc("alphacorp", "Alpha Corp", 2022, "4,220", "1,520", "Alpha Steel Mills"),
            company_doc("alphacorp", "Alpha Corp", 2023, "4,330", "1,630", "Alpha Steel Mills"),
            company_doc("betaltd", "Beta Ltd", 2022, "2,120", "870", "Beta Rolling Plant", style="employees"),
            company_doc("betaltd", "Beta Ltd", 2023, "2,230", "910", "Beta Rolling Plant", style="employees"),
            other_doc("gammainc", "Gamma Inc", "container shipping"),
            other_doc("deltaco", "Delta Co", "grocery retail"),
            other_doc("epsilonplc", "Epsilon Plc", "hotel services"),
            other_doc("zetagroup", "Zeta Group", "cement trading"),
            other_doc("etaholding", "Eta Holding", "software licences")]
    return CorpusSet("T_en", docs)


ALPHA22, BETA22 = "T.alphacorp_annual-report_2022.en", "T.betaltd_annual-report_2022.en"


class ScriptClient(FakeOllama):
    """A real OllamaClient (ledger, options) with scripted chat replies by purpose (a string, an object, a callable of
    the messages, or an Exception to raise) and every embed purpose recorded."""

    def __init__(self, cfg, replies):
        super().__init__(cfg, Ledger(path=None))
        self.replies = replies
        self.purposes, self.embed_purposes = [], []
        self.messages = {}

    def chat(self, messages, *, purpose, **kw):
        self.purposes.append(purpose)
        self.messages[purpose] = messages
        rep = self.replies.get(purpose, "")
        if isinstance(rep, Exception):
            raise rep
        if callable(rep):
            rep = rep(messages)
        self._next = rep if isinstance(rep, str) else json.dumps(rep, ensure_ascii=False)
        if not self.ledger.context().get("phase"):
            kw.setdefault("phase", "query")
        return super().chat(messages, purpose=purpose, **kw)

    def _post(self, endpoint, payload):
        if endpoint == "/api/chat":
            return ({"message": {"content": self._next}, "prompt_eval_count": 10, "eval_count": 3,
                     "done_reason": "stop"}, 1.0, 0.0, 1)
        return super()._post(endpoint, payload)

    def embed(self, texts, *, purpose, **kw):
        self.embed_purposes.append(purpose)
        if not self.ledger.context().get("phase"):
            kw.setdefault("phase", "query")
        return super().embed(texts, purpose=purpose, **kw)


def picker(*words):
    """A call-2 responder: the IDs of the listed phrases containing any of `words`, for every facet."""
    def reply(messages):
        user = messages[-1]["content"]
        ids = [m.group(1) for m in re.finditer(r"^(P\d+): (.*?) \[", user, re.M)
               if any(w in m.group(2).lower() for w in words)]
        nf = len(re.findall(r"^\d+: ", user.split("Phrases found")[0], re.M))
        return {"facets": [{"facet": i, "phrases": ids[:4], "add": ["payroll"]} for i in range(1, nf + 1)]}
    return reply


PLAN_CMP = {"entities": ["Alpha Corp", "Beta Ltd"], "periods": ["2022"], "operation": "compare",
            "facets": [{"name": "revenue", "terms": ["revenue", "sales"]}], "answer": "number"}
PLAN_STAFF = {"entities": ["Alpha Corp"], "periods": ["2022"], "operation": "lookup",
              "facets": [{"name": "workforce", "terms": ["workforce", "number of workers"]}], "answer": "number"}


def query(text, scope, qid="q"):
    return Query(question_id=qid, text=text, q_lang="en", d_lang="en", setting="en-en", corpus="T", corpus_set="T_en",
                 doc_scope=scope, answer_format="default")


def agent(cfg, replies, flags=None, options=None):
    client = ScriptClient(cfg, replies)
    ap = Lex3Agent(cfg=cfg, client=client, flags=flags, options=dict({"call_cache": "off"}, **(options or {})))
    return ap, client


# ---------------------------------------------------------------------------------------------------------------
# 1. prompts
# ---------------------------------------------------------------------------------------------------------------

def prompt_tests():
    print("\n[prompts]")
    o = dict(DEFAULTS)
    it, pr, how = P.validate_interpretation(json.dumps(dict(PLAN_CMP, periods=["۱۳۹۸ تا ۱۴۰۰", "2022"])), o)
    check(how == "direct" and it["entities"] == ["Alpha Corp", "Beta Ltd"] and it["periods"] == ["1398", "1400", "2022"]
          and it["operation"] == "compare" and not it["empty"] and not pr,
          "a valid interpretation passes (Persian digits read)", (it, pr))
    it, pr, _h = P.validate_interpretation('Sure! {"facets": [{"name": "a", "terms": ["x1"]}, {"name": "b", "terms": '
                                           '["x2"]}, {"name": "c", "terms": ["x3"]}, {"name": "d", "terms": ["x4"]}], '
                                           '"operation": "guess", "entities": [], "periods": [], "answer": "number"}', o)
    check(len(it["facets"]) == 3 and it["operation"] == "other" and any("capped" in p for p in pr)
          and any("operation" in p for p in pr), "facets capped at 3, a bad operation reported", pr)
    it, pr, how = P.validate_interpretation("I cannot help with that.", o)
    check(it["empty"] and how == "failed" and "no JSON" in pr, "garbage -> an EMPTY plan, and why")
    it, pr, _h = P.validate_interpretation('{"facets": [{"name": "", "terms": []}], "entities": [], "periods": [], '
                                           '"operation": "lookup", "answer": "text"}', o)
    check(it["empty"] and any("no usable term" in p for p in pr), "a facet without a term is dropped -> empty")
    it, _pr, _h = P.validate_interpretation('{"keywords": ["revenue"], "names": ["Alpha"], "years": ["2021"]}', o)
    check(it["facets"] and it["facets"][0]["terms"] == ["revenue"] and it["entities"] == ["Alpha"],
          "LEX-shaped keywords are accepted as one facet (recorded)")
    vocab = [{"id": "P1", "text": "Revenue"}, {"id": "P2", "text": "Staff"}]
    got, pr, _h = P.validate_pick('{"facets": [{"facet": 1, "phrases": ["P2", "P9", 1], "add": ["pay roll"]}, '
                                  '{"facet": 7, "phrases": ["P1"], "add": []}]}', 1, vocab, o)
    check(got["by_facet"] == {0: {"phrases": ["P2", "P1"], "add": ["pay roll"]}} and any("P9" in p for p in pr)
          and any("out of range" in p for p in pr), "unknown IDs and out-of-range facets are reported, not used",
          (got, pr))
    got, _pr, _h = P.validate_pick('{"pick": ["P1"]}', 2, vocab, o)
    check(got["shape"] == "flat" and set(got["by_facet"]) == {0, 1}, "a flat pick applies to every facet")
    got, pr, how = P.validate_pick("no", 1, vocab, o)
    check(how == "failed" and got["shape"] == "none", "an unusable pick reply is recognised as such")


# ---------------------------------------------------------------------------------------------------------------
# 2. lexical refinements and fusion (no index needed)
# ---------------------------------------------------------------------------------------------------------------

def lexical_tests(cs):
    print("\n[lexical]")
    o = dict(DEFAULTS)
    lit = S.compile_terms(["produced"], "en", "picked", 1.0, o)
    mt = C.morph_terms(lit, "en", o)
    check(len(mt) == 1 and mt[0].weight < lit[0].weight, "a morph twin exists and weighs less (%s)" % [m.weight for m in mt])
    check(lit[0].search(fold("output was producing more")) < 0 and mt[0].search(fold("output was producing more")) >= 0,
          "morphology adds a match the literal term misses (produced ~ producing)")
    check(mt[0].search(fold("the plant produced steel")) < 0 and lit[0].search(fold("the plant produced steel")) >= 0,
          "where the literal term matches, the morph twin does not (no double credit)")
    check(mt[0].search(fold("production rose")) < 0, "no derivational strip: produced !~ production")
    lit[0].idf, mt[0].idf = 2.0, 9.0
    C.align_morph_idf(lit + mt)
    check(mt[0].idf == 2.0, "a morph twin takes its literal term's IDF")

    heads = {ALPHA22: "alpha corp annual report 2022 | alphacorp", BETA22: "beta ltd annual report 2022 | betaltd"}
    wins = [{"doc_id": ALPHA22, "hit": "a", "score": 1.0, "text": "Alpha Corp staff at Alpha Steel Mills"},
            {"doc_id": ALPHA22, "hit": "b", "score": 1.0, "text": "Alpha Corp staff at year end"},
            {"doc_id": BETA22, "hit": "c", "score": 1.0, "text": "unlike Alpha Corp, Beta keeps staff"}]
    C.name_bonus(wins, ["Alpha Corp", "Alpha Steel Mills"], heads, o)
    by = {w["hit"]: w for w in wins}
    check(by["b"]["name_bonus"][0] == "issuer" and abs(by["b"]["score"] - (1 + o["w_name"] * o["w_issuer_name"])) < 1e-6,
          "the issuer's own name inside its own report is demoted", by["b"])
    check(by["a"]["name_bonus"] == ["name", o["w_name"]], "a subsidiary/plant name in the issuer's report keeps full weight",
          by["a"])
    check(by["c"]["name_bonus"] == ["name", o["w_name"]], "the same company name in ANOTHER company's report keeps full weight")
    check(C.is_issuer("Alpha", heads[ALPHA22]) and not C.is_issuer("Alpha Steel Mills", heads[ALPHA22]),
          "issuer test: part of the title yes, a longer subsidiary name no")

    print("\n[fusion]")
    doc = cs.doc(ALPHA22)
    u = doc.unit(ALPHA22 + "::people#p0")
    a0, a1 = u.span
    ch = {"bm25": [{"doc_id": ALPHA22, "span": [a0, a1]}, {"doc_id": ALPHA22, "span": [a0 + 5, a1]},
                   {"doc_id": BETA22, "span": list(cs.doc(BETA22).unit(BETA22 + "::fin#p0").span)}],
          "lex": [{"doc_id": ALPHA22, "span": None, "unit_ids": [u.id]}]}
    regs, info = C.fuse_regions(ch, cs, o)
    top = regs[0]
    check(top["doc_id"] == ALPHA22 and len(top["members"]) == 3 and top["ranks"] == {"bm25": 1, "lex": 1}
          and abs(top["rrf"] - (1 / 61 + 1 / 61)) < 1e-12,
          "overlapping candidates form ONE region; one contribution per channel (%s)" % top["ranks"])
    check(abs(regs[1]["rrf"] - 1 / 62) < 1e-12, "the next region takes the channel's next rank (2), not 3")
    regs2, _i = C.fuse_regions(dict(reversed(list(ch.items()))), cs, o)
    check([(r["rid"], round(r["rrf"], 12)) for r in regs] == [(r["rid"], round(r["rrf"], 12)) for r in regs2],
          "fusion is independent of channel order")
    cands = C.block_candidates(regs, 10)
    check(cands[0]["rrf"] == top["rrf"] and cands[0]["region"] == top["rid"], "block candidates carry region RRF")
    reqs, rinfo = C.requirements([{"name": "rev", "terms": ["revenue"]}], ["Alpha Corp", "Beta Ltd"], ["2022"],
                                 [["revenue", "Revenue by segment"]], o)
    check([r["id"] for r in reqs] == ["F1|e:Alpha Corp", "F1|e:Beta Ltd"] and all(r["terms"] == ["revenue", "Revenue by segment"]
                                                                                   for r in reqs),
          "requirements: facet x entity (a comparison), the same vocabulary for both", reqs)
    reqs, rinfo = C.requirements([{"name": "a", "terms": ["x"]}, {"name": "b", "terms": ["y"]}], ["A", "B"],
                                 ["2019", "2020", "2021"], [["x"], ["y"]], o)
    check(len(reqs) <= o["max_requirements"] and rinfo["merged_facets"], "too many requirements: facets merged first",
          (len(reqs), rinfo))


# ---------------------------------------------------------------------------------------------------------------
# 3. with the index: vocabulary, routing, sessions
# ---------------------------------------------------------------------------------------------------------------

def index_tests(cfg, cs):
    print("\n[index]")
    ap, client = agent(cfg, {})
    index = ap.build_index(cs, client.ledger)
    hx = index.payload["hybrid"]
    check(hx.dense_complete and set(client.embed_purposes) == {"index-embed"}, "index time: every chunk embedded "
          "(index-embed only)", client.embed_purposes[:3])
    return index


def vocab_tests(cfg, cs, index):
    print("\n[vocab]")
    o = dict(DEFAULTS)
    hx = index.payload["hybrid"]
    client = ScriptClient(cfg, {})
    q = "How large was the workforce of Alpha Corp in 2022?"
    hits = hx.search(q, doc_ids=cs.doc_ids(), k=80, depth=40, client=client)
    facets = [{"name": "workforce", "terms": ["workforce", "number of workers"]}]
    vocab, info = C.feedback_vocabulary(hits, cs, facets, q, "en", o)
    ok = all(v["text"] in cs.doc(v["source"]["doc_id"]).text for v in vocab)
    check(vocab and ok, "every phrase is an exact string of its source report (%d phrases)" % len(vocab),
          [v for v in vocab if v["text"] not in cs.doc(v["source"]["doc_id"]).text][:3])
    check(len({v["id"] for v in vocab}) == len(vocab) and all(re.fullmatch(r"P\d+", v["id"]) for v in vocab),
          "phrase IDs unique")
    check(all(v["source"]["doc_id"] in {h.doc_id for h in hits} and v["source"]["hit_rank"] >= 1 for v in vocab),
          "each phrase names the retrieved hit it came from")
    kinds = {v["kind"] for v in vocab}
    check({"heading", "caption", "label"} <= kinds, "headings, captions and table labels all reach the list: %s" % kinds)
    check(any(fold(v["text"]) == "headcount by plant" for v in vocab), "the caption prefix «Table 1:» is stripped")
    check(info["docs"] >= 2 and len(vocab) <= o["vocab_max"], "spread over several reports, bounded (%d docs)" % info["docs"])
    return hits


def route_tests(cfg, cs, index, hits):
    print("\n[routing]")
    o = dict(DEFAULTS, shortlist_max=3)
    ap, _c = agent(cfg, {})
    con = ap.kb()
    pool = cs.doc_ids()
    kw = {"keywords": ["workforce", "staff"], "names": ["Alpha Corp"], "years": ["2022"]}
    forced = C.explicit_matches(con, pool, ["Beta Ltd"], ["2023"])
    check(forced == ["T.betaltd_annual-report_2023.en"], "explicit entity+period match found: %s" % forced)
    short, info = C.route(con, pool, hits, kw, "en", o, forced=forced)
    check(len(short) == 3 and short[0] == forced[0], "the forced report is kept first in a 3-report shortlist: %s" % short)
    check(ALPHA22 in short, "the card + capped chunk score finds the asked report")
    cc = C.chunk_contribution(hits, 2)
    per_doc = {}
    for h in hits:
        per_doc.setdefault(h.doc_id, []).append(h.rrf)
    check(all(abs(cc[d] - sum(sorted(v, reverse=True)[:2])) < 1e-12 for d, v in per_doc.items()),
          "a report's chunk contribution is its best two chunks, not the sum of all")


def session_tests(cfg, cs, index):
    print("\n[session]")
    budget = int(cfg.evidence["max_chars"])
    scope = cs.doc_ids()

    # the full path: interpret, pick, answer
    ap, cl = agent(cfg, {"plan": PLAN_STAFF, "pick": picker("staff", "headcount"), "answer": "4,220 staff"})
    q = query("How large was the workforce of Alpha Corp in 2022?", scope, "staff")
    res = _Session3(ap, q, index).run()
    tr = res.trace
    check(cl.purposes == ["plan", "pick", "answer"] and tr["degraded_mode"] == "none",
          "3 calls: interpret, pick, answer (%s)" % cl.purposes)
    terms = tr["stages"]["lexical"]["terms"]
    picked = [t for t in terms if (t.get("provenance") or {}).get("source") == "picked"]
    vocab = {v["id"]: v for v in tr["stages"]["vocab"]["phrases"]}
    check(picked and all(t["provenance"]["phrase_id"] in vocab and t["provenance"]["observed"]
                         and vocab[t["provenance"]["phrase_id"]]["text"] == t["text"] for t in picked),
          "picked terms carry their phrase ID, observed, and the phrase's text", picked[:2])
    added = [t for t in terms if (t.get("provenance") or {}).get("source") == "added"]
    check(added and all(not t["provenance"]["observed"] for t in added), "added terms are marked NOT observed")
    check(any(t.get("morph_of") for t in terms), "morph twins join the literal terms")
    check("lex" in tr["stages"]["fuse"]["by_channel"] and {"bm25", "dense"} <= set(tr["stages"]["fuse"]["by_channel"]),
          "the lexical channel fuses with the question's BM25 and dense channels")
    check(any("4,220" in it.text for it in res.evidence.items), "the answer passage is in the evidence")
    check(res.evidence.chars <= budget and not res.evidence.last_item_cut and res.evidence.n_included == len(res.retrieved),
          "exact budget: every block fits whole (%d chars, %d items)" % (res.evidence.chars, res.evidence.n_included))
    rendered = pack_evidence(res.retrieved, cs, budget=budget, cfg=cfg)
    check(rendered.chars == tr["stages"]["blocks"].get("rendered_chars") == res.evidence.chars,
          "the blocks' own count equals the rendered evidence (%s / %d)" % (tr["stages"]["blocks"].get("rendered_chars"),
                                                                            rendered.chars))
    check(len(tr["stages"]["route"]["shortlist"]) <= 8 and tr["stages"]["route"]["routed"],
          "ten reports: routed to a shortlist of <= 8 (%d)" % len(tr["stages"]["route"]["shortlist"]))
    check(set(cl.embed_purposes) <= {"query-embed"} and cl.embed_purposes,
          "no document embedding at query time (embeds: %s)" % sorted(set(cl.embed_purposes)))

    # a comparison: one block reserved per requirement (per company)
    ap, cl = agent(cfg, {"plan": PLAN_CMP, "pick": picker("revenue"), "answer": "x"})
    q = query("Compare the revenue of Alpha Corp and Beta Ltd in 2022.", scope, "cmp")
    res = _Session3(ap, q, index).run()
    tr = res.trace
    rq = tr["stages"]["requirements"]
    check([r["id"] for r in rq["list"]] == ["F1|e:Alpha Corp", "F1|e:Beta Ltd"],
          "requirements from the interpretation: one per company", rq["list"])
    check(all((rq["status"].get(r["id"]) or {}).get("status") == "covered" for r in rq["list"]),
          "both requirements covered by a reserved block", rq["status"])
    marked = [it for it in res.retrieved if (it.meta or {}).get("reserved")]
    check(len(marked) >= 2 and all(it.meta.get("requirement") for it in marked),
          "reserved items are marked for the variants-v2 section expansion (meta.reserved / meta.requirement)")
    check({ALPHA22, BETA22} <= {it.doc_id for it in res.evidence.items}, "both companies' 2022 reports in the evidence")
    reqterms = rq["list"][0]["terms"]
    check("revenue" in reqterms and any(t != "revenue" and t != "sales" for t in reqterms),
          "a requirement's vocabulary = facet terms + the picked source phrases", reqterms)

    # empty-plan injection: the question channels answer, with the full hybrid budget
    ap, cl = agent(cfg, {"plan": "{}", "answer": "x"})
    q = query("How large was the workforce of Alpha Corp in 2022?", scope, "empty")
    res = _Session3(ap, q, index).run()
    tr = res.trace
    check(cl.purposes == ["plan", "answer"] and tr["degraded_mode"] == "empty_plan",
          "empty plan: no call 2, degraded_mode recorded (%s, %s)" % (cl.purposes, tr["degraded_mode"]))
    check(set(tr["stages"]["fuse"]["by_channel"]) == {"bm25", "dense"} and res.evidence.n_included >= 3,
          "... answered from the question's BM25 + dense channels (%d blocks)" % res.evidence.n_included)
    dropped = {d.get("reason") for d in tr["stages"]["blocks"].get("dropped") or []}
    check(res.evidence.chars > 1200 and not tr["stages"]["lexical"]["terms"] and "budget" not in dropped
          and res.evidence.n_included == DEFAULTS_EVIDENCE_CAP,
          "... up to the block cap with no source cap, not LEX2's 1,200-char safety net (%d chars, %d blocks)"
          % (res.evidence.chars, res.evidence.n_included))
    ap, cl = agent(cfg, {"plan": ModelError("down"), "answer": "x"})
    res = _Session3(ap, q, index).run()
    check(res.trace["degraded_mode"] == "empty_plan" and "failed" in res.trace["degraded_reason"]
          and cl.purposes == ["plan", "answer"] and res.evidence.n_included > 0,
          "a FAILED interpret call is the same mode, with its reason")

    # call 2 fails: keyword (facet terms) + question retrieval kept
    for label, rep in (("raises", ModelError("pick down")), ("garbage", "no idea"), ("empty", {"facets": []})):
        ap, cl = agent(cfg, {"plan": PLAN_STAFF, "pick": rep, "answer": "x"})
        q = query("How large was the workforce of Alpha Corp in 2022?", scope, "pickfail-" + label)
        res = _Session3(ap, q, index).run()
        tr = res.trace
        srcs = {(t.get("provenance") or {}).get("source") for t in tr["stages"]["lexical"]["terms"]}
        mode = tr["degraded_mode"]
        check(cl.purposes == ["plan", "pick", "answer"] and mode in ("pick_failed", "pick_empty") and "keyword" in srcs
              and {"bm25", "dense"} <= set(tr["stages"]["fuse"]["by_channel"]) and res.evidence.n_included > 0,
              "call 2 %s: %s, keyword + question channels kept (%s)" % (label, mode, sorted(x for x in srcs if x)))
        kw_w = [t["weight"] for t in tr["stages"]["lexical"]["terms"] if (t.get("provenance") or {}).get("source") == "keyword"]
        check(kw_w and all(w == 1.0 for w in kw_w), "... the facet terms then ARE the lexical channel (weight 1)")

    # no usable phrase list: skip call 2
    plan = dict(PLAN_STAFF, facets=[{"name": "staff", "terms": ["staff", "headcount"]}])
    ap, cl = agent(cfg, {"plan": plan, "answer": "x"}, options={"vocab_max": 0})
    res = _Session3(ap, query("How large was the workforce of Alpha Corp in 2022?", scope, "novocab"), index).run()
    check(cl.purposes == ["plan", "answer"] and res.trace["degraded_mode"] == "no_shortlist"
          and "lex" in res.trace["stages"]["fuse"]["by_channel"], "no phrase list: call 2 skipped, facet terms kept")

    # the call cap: with 2 calls allowed the answer is still made (call 2 is skipped, not the answer)
    ap, cl = agent(cfg, {"plan": PLAN_STAFF, "pick": picker("staff"), "answer": "x"})
    ap.max_calls = 2
    res = _Session3(ap, query("How large was the workforce of Alpha Corp in 2022?", scope, "cap"), index).run()
    check(cl.purposes == ["plan", "answer"] and res.trace["stages"]["pick"].get("skipped"),
          "a 2-call cap skips call 2 and keeps the answer (%s)" % cl.purposes)
    check(max(len(x) for x in [cl.purposes]) <= 3 and ap.max_calls == 2 and Lex3Agent(cfg=cfg, client=cl).max_calls == 3,
          "the default cap is 3 (caps.lex_max_calls)")

    # the escape channel: a shortlist of 1 cannot lose the other company's report
    ap, cl = agent(cfg, {"plan": PLAN_CMP, "pick": picker("revenue"), "answer": "x"},
                   options={"shortlist_max": 1, "forced_max": 1, "followup_max": 0})
    q = query("Compare the revenue of Alpha Corp and Beta Ltd in 2022.", scope, "escape")
    res = _Session3(ap, q, index).run()
    tr = res.trace
    short = tr["stages"]["route"]["shortlist"]
    esc = tr["evidence"]["escape_docs"]
    check(len(short) == 1 and esc and tr["stages"]["fuse"]["escape_regions"] > 0,
          "question candidates outside a 1-report shortlist reach the evidence (escape: %s)" % esc)
    ap, cl = agent(cfg, {"plan": PLAN_CMP, "pick": picker("revenue"), "answer": "x"}, flags={"no_escape": True},
                   options={"shortlist_max": 1, "forced_max": 1, "followup_max": 0})
    res = _Session3(ap, q, index).run()
    check(not res.trace["evidence"]["escape_docs"] and res.trace["stages"]["fuse"]["escape_regions"] == 0,
          "no_escape drops them (the routing-without-escape control)")

    # follow-up: a missing requirement triggers one bounded retrieval over the whole pool
    ap, cl = agent(cfg, {"plan": PLAN_CMP, "pick": picker("revenue"), "answer": "x"}, flags={"no_escape": True},
                   options={"shortlist_max": 1, "forced_max": 1})
    res = _Session3(ap, q, index).run()
    fu = res.trace["stages"]["requirements"]["followup"]
    check(fu and len(fu) <= DEFAULTS["followup_max"] and any(f["new_candidates"] for f in fu),
          "a missing requirement triggers a bounded follow-up (%s)" % [(f["requirement"], f["new_candidates"]) for f in fu])
    check(set(cl.embed_purposes) <= {"query-embed"}, "follow-up embeds only its query")

    # variants card hook and the doc_card option (planning calls get the card; the cache key follows the request)
    ap, cl = agent(cfg, {"plan": PLAN_STAFF, "pick": picker("staff"), "answer": "x"}, flags={"doc_card": True})
    ap.variants_version = "variants-v2"
    q = query("How large was the workforce of Alpha Corp in 2022?", scope, "card")
    ids = ap.variant_routed_ids(q, index)
    check(ids and len(ids) <= 8 and ids[0].startswith("T.alphacorp"), "the card hook gets the question-only shortlist: %s"
          % (ids or [])[:3])
    ap._begin_variants(q, index)
    res = _Session3(ap, q, index).run()
    sel = ap._variant_trace["card_selection"]
    check(sel["routed_ids"] == ids and cl.messages["plan"][-1]["content"].startswith("Context")
          and cl.messages["pick"][-1]["content"].startswith("Context")
          and not cl.messages["answer"][-1]["content"].startswith("Context"),
          "doc_card: both planning calls see the card, the answer call does not")
    n_embed = cl.embed_purposes.count("query-embed")
    check(n_embed >= 1 and res.trace["stages"]["question"]["hits"] > 0, "one shared question search for hook + session")


def flag_tests(cfg):
    print("\n[flags]")
    check(parse_flags("no_pick,no_morph") == {"no_pick": True, "no_morph": True}, "flags parse")
    for bad in ("window=300", "no_lexical,no_pick", "bm25"):
        try:
            parse_flags(bad)
            check(False, "refused: %s" % bad)
        except ValueError:
            check(True, "refused: %s" % bad)
    from ...approaches import APPROACHES, LEX3_FLAGS, get_approach
    from ...approaches.lex.defaults import UPGRADES
    from ...approaches.lex import Lex2Agent
    from ... import run as R
    check(APPROACHES["LEX3"].endswith(":Lex3Agent") and R.parse_flags(["no_pick"], "LEX3") == {"no_pick": True}
          and tuple(LEX3_FLAGS) == tuple(__import__("ragbench.approaches.lex3.defaults", fromlist=["FLAGS"]).FLAGS),
          "registered; run.py accepts LEX3 flags")
    check(Lex2Agent.ON == UPGRADES == ("bm25", "report_vocab", "year_pass"), "LEX2's upgrade set is unchanged")
    a = get_approach("LEX3", cfg=cfg, client=ScriptClient(cfg, {}), flags={"no_followup": True})
    check(a.variant == "LEX3+no_followup", "variant name %s" % a.variant)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cfg = load_config().copy_with({"paths": {"index_root": str(TMP / "indexes")}, "kb": {"path": str(TMP / "kb.sqlite")},
                                   "ollama": {"cross_process_lock": False}})
    cs = corpus()
    prompt_tests()
    lexical_tests(cs)
    index = index_tests(cfg, cs)
    hits = vocab_tests(cfg, cs, index)
    route_tests(cfg, cs, index, hits)
    session_tests(cfg, cs, index)
    flag_tests(cfg)
    print("\n%d checks, %d failure(s)" % (N[0], len(FAILS)))
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
