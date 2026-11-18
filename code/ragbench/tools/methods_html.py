"""how_methods_work.html: every method of the study drawn as a simple flow diagram - how a question moves through it -
with one real question as the running example. Plain page in the project folder, no framework, no script.

    .venv/Scripts/python.exe -m ragbench.tools.methods_html

The diagrams are inline SVG built from METHODS below. The top lane is what a method prepares BEFORE any question; the
bottom lane is what happens for EACH question. A filled box is a call to the 4B model, so counting the filled boxes of
a diagram counts its model calls (a box marked "repeats" may be called several times). Each diagram is drawn twice -
one row per lane for wide screens, three boxes per row for phones - and CSS shows one of them.

The example is the DEV timeline question C-TL-divar-01 ("how many people did Divar employ each year from 1397 to
1400, and how did that change?"). Its gold answer is NOT comparable across years - four years, four definitions of
"employee" - so a correct answer must give the values AND refuse to state a trend, the behaviour that separates the
methods most. Every method answered it with the same model, the same answer prompt and the same 6000-character
evidence budget. The result strip of each method is read from its own run record and scored with the study's own
timeline scorer (eval/timeline.py, text scoring), so the marks on this page are the marks in the paper.
"""
import html
import json
import re
import sys
from pathlib import Path

from ..contracts import load_questions
from ..eval import timeline as TL

ROOT = Path(__file__).resolve().parents[2]
QID = "C-TL-divar-01"
QFILE = ROOT / "ragbench/data/questions/pilot_timeline.jsonl"
OUT = ROOT / "how_methods_work.html"
R = ROOT / "work/runs"


def n(kind, text, tag=None, rep=None, new=False):
    """One box of a diagram. kind: q question · m model call · v embedding (small vector model) · s plain code ·
    d stored before any question · e evidence handed to the model · a the answer (a model call)."""
    return {"k": kind, "t": text, "tag": tag, "rep": rep, "new": new}


Q = n("q", "Question")
ANSWER = n("a", "Model writes\nthe answer")
CHECKED = n("e", "Checked\ntable")

GROUPS = [
    ("Baselines", "The usual ways to give a model documents, plus two reference points."),
    ("Our regex family", "The model writes a few search patterns; plain code runs them over the reports."),
    ("Our agents", "Plan the search, check every number against its quote, line the numbers up by year."),
]

# uses = the index of the per-question box that reads what was prepared (the dashed arrow between the lanes)
METHODS = [
    dict(v="B0", g=0, name="B0 · No search", dirs=["html_example"],
         line="No documents at all: the model answers from memory.",
         prep=[], run=[Q, n("a", "Model answers\nfrom memory")], uses=None),
    dict(v="B1", g=0, name="B1 · The right passages (a cheat sheet)", dirs=["html_example"],
         line="The model is handed the passages a person marked as the answer: the best any search could do.",
         prep=[n("d", "Passages\nmarked by\na person")], run=[Q, n("e", "The right\npassages"), ANSWER], uses=1),
    dict(v="B2", g=0, name="B2 · Keyword search", dirs=["html_example"],
         line="Picks the text chunks that share the most words with the question.",
         prep=[n("s", "Cut reports\ninto chunks"), n("d", "Chunks")],
         run=[Q, n("s", "Match the\nrare words", tag="BM25"), n("e", "Best\nchunks"), ANSWER], uses=1),
    dict(v="B3", g=0, name="B3 · Meaning search", dirs=["core_dev_v4_tl", "core_dev_pilot_timeline"],
         line="Text and question become number vectors; the chunks closest in meaning are picked.",
         prep=[n("s", "Cut reports\ninto chunks"), n("v", "Each chunk\nto a vector"), n("d", "Vector\nindex")],
         run=[Q, n("v", "Question\nto a vector"), n("s", "Find the\nclosest chunks"), n("e", "Best\nchunks"), ANSWER],
         uses=2),
    dict(v="B4", g=0, name="B4 · Table-of-contents walk", dirs=["html_example"],
         line="Like reading a book's contents: the model picks a report, then a chapter, then a section.",
         prep=[n("m", "Model sums up\nevery section"), n("d", "Contents +\nsummaries")],
         run=[Q, n("m", "Pick the\nreports"), n("m", "Walk down\nthe contents", rep="↻ each level"),
              n("e", "Chosen\nsections"), ANSWER], uses=1),
    dict(v="B5", g=0, name="B5 · Search agent", dirs=["html_example"],
         line="Like a coding agent: the model writes a text search, reads the hits, and searches again.",
         prep=[], run=[Q, n("m", "Write a\nsearch", rep="↻ up to 7×"), n("s", "Run it,\nshow the hits", tag="grep",
                                                                         rep="↻"),
                       n("e", "Sections\nit read"), ANSWER], uses=None),
    dict(v="OURS+index_level=L0", g=1, name="L0 · Search patterns", dirs=["html_example"],
         line="The model writes a few search patterns; plain code runs them over every section and number.",
         prep=[n("s", "Find numbers\nwith regex", tag="regex"), n("d", "Sections +\nnumbers")],
         run=[Q, n("m", "Write search\npatterns"), n("s", "Run the\npatterns", tag="regex"),
              n("e", "Matching\ntext"), ANSWER], uses=2),
    dict(v="OURS+index_level=L1", g=1, name="L1 · Patterns + a glossary", dirs=["html_example"],
         line="As L0, plus a small glossary per report, so a search also finds the report's own words.",
         prep=[n("s", "Find numbers\nwith regex", tag="regex"), n("m", "Glossary for\neach report"),
               n("d", "Sections +\nglossary")],
         run=[Q, n("m", "Write search\npatterns"), n("s", "Run patterns\n+ glossary", tag="regex"),
              n("e", "Matching\ntext"), ANSWER], uses=2),
    dict(v="OURS", g=1, name="L3 · Patterns over annotated reports", dirs=[],
         line="As L0, but the model first annotates every section of every report. Costly: hours of model time.",
         prep=[n("m", "Model notes\nevery section", rep="↻ hours"), n("d", "Annotated\nsections")],
         run=[Q, n("m", "Write search\npatterns"), n("s", "Run the\npatterns", tag="regex"),
              n("e", "Matching\ntext"), ANSWER], uses=2),
    dict(v="LEX", g=1, name="LEX · Cards, the reports' own words, strong regex", dirs=["lex_dev_tl"],
         line="The model writes keywords; each report's card (its name and keywords) picks the reports; the model "
              "picks the reports' own words for what is asked; code turns them into regexes, searches the whole "
              "reports and cuts a window around every hit.",
         prep=[n("s", "Load reports\ninto a\ndatabase", tag="no model"),
               n("s", "A card per\nreport: name +\nkeywords", tag="no model", new=True), n("d", "Cards +\nfull text")],
         run=[Q, n("m", "Write\nkeywords"), n("s", "Cards pick\nthe reports", tag="cards", new=True),
              n("m", "Pick the\nreports' own\nwords"), n("s", "Regex over\nwhole reports", tag="regex"),
              n("e", "A window\naround each\nhit"), ANSWER], uses=2),
    dict(v="AGENTIC", g=2, name="Agent v1 · Plan, check, line up", dirs=["core_dev_v4_tl", "core_dev_pilot_timeline"],
         line="Plans the search, checks every number against its quote, and lines the numbers up by year.",
         prep=[], run=[Q, n("m", "Plan: words,\nyears,\nreports"), n("s", "Cut text\nwindows", tag="regex"),
                       n("m", "Check each\nnumber vs\nits quote"), n("s", "Line up\nby year", tag="code"),
                       CHECKED, ANSWER], uses=None),
    dict(v="AGENTIC2", g=2, name="Agent v2 · A wider search", dirs=["core_dev_v4_tl", "core_dev_pilot_timeline"],
         line="v1 with a wider search: words, numbers and meaning together, in only the reports it needs.",
         prep=[], prep_note="Nothing in advance: a report's vectors are made the first time it is searched.",
         run=[Q, n("m", "Plan"), n("s", "Search: words,\nnumbers,\nmeaning", tag="+ vectors"),
              n("m", "Check each\nnumber vs\nits quote"), n("s", "Line up\nby year", tag="code"), CHECKED, ANSWER],
         uses=None),
    dict(v="AGENTIC3", g=2, name="Agent v3 · A database of numbers", dirs=["core_dev_v4_tl", "core_dev_pilot_timeline"],
         line="Every report is first loaded into a database of numbers with their unit, year and definition.",
         prep=[n("s", "Load reports\ninto a\ndatabase", tag="no model"), n("d", "Numbers with\nunit + year")],
         run=[Q, n("m", "Plan"), n("s", "Search the\ndatabase", tag="SQL"), n("m", "Check each\nnumber vs\nits quote"),
              n("s", "Line up by\nyear, flag\nchanges", tag="SQL"), CHECKED, ANSWER], uses=2),
    dict(v="AGENTIC4", g=2, name="Agent v4 · v3 + a graph for missing years", dirs=["core_dev_v4_tl"],
         line="v3, plus a graph of metric names: when a year is still missing, it looks for the same metric in that "
              "year's report, even when the report names it differently.",
         prep=[n("s", "Load reports\ninto a\ndatabase", tag="no model"),
               n("s", "Link metric\nnames across\nyears", tag="graph", new=True), n("d", "Numbers +\nthe graph")],
         run=[Q, n("m", "Plan"), n("s", "Search the\ndatabase", tag="SQL"),
              n("m", "Check each\nnumber vs\nits quote"), n("s", "Line up by\nyear, flag\nchanges", tag="SQL"),
              n("s", "Year missing?\nThe graph\nlooks there", tag="graph", new=True), CHECKED, ANSWER], uses=2),
]

# ---- diagram geometry (SVG user units) -----------------------------------------------------------------------------
NH = 58                      # box height
LH = 13.5                    # line height of box text
WIDE = dict(width=760, nw=90, gap=20, cols=7, row_gap=32)
NARROW = dict(width=360, nw=98, gap=27, cols=3, row_gap=34)

CSS = """
:root { --bg:#f6f6f3; --card:#ffffff; --ink:#1e2630; --muted:#5f6b76; --line:#dde1e6; --line-strong:#98a3ae;
        --accent:#2b59c3; --on-accent:#ffffff; --accent-soft:#e3eafb; --store:#eef0f3; --q:#f1f3f5;
        --ok:#1d6f35; --ok-bg:#e2f3e6; --no:#a1262b; --no-bg:#fbe6e7; --chip:#eef0f3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#13161b; --card:#1b1f26; --ink:#e5e8ed; --muted:#9aa4b0; --line:#2c323b; --line-strong:#5d6875;
          --accent:#7196ee; --on-accent:#0c1220; --accent-soft:#1e2944; --store:#232833; --q:#232833;
          --ok:#7fd49a; --ok-bg:#16301f; --no:#f19a9e; --no-bg:#3a1a1d; --chip:#262b34; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); line-height: 1.5;
       font-family: "Segoe UI", system-ui, -apple-system, "Vazirmatn", Tahoma, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 18px; max-width: 64ch; }
h2 { font-size: 1.15rem; margin: 34px 0 2px; }
.gline { color: var(--muted); margin: 0 0 10px; font-size: 0.95rem; }
.fa { font-family: "Vazirmatn", Tahoma, "Segoe UI", sans-serif; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 10px 14px; font-size: 0.86rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.legend svg { width: 40px; height: 26px; flex: none; }
.example { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
           margin-top: 16px; }
.example .q-fa { font-size: 1.05rem; margin: 0 0 2px; }
.example .q-en { color: var(--muted); margin: 0 0 12px; font-size: 0.93rem; }
.strip { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.yr { background: var(--accent-soft); border-radius: 7px; padding: 3px 10px; font-size: 0.9rem; }
.yr b { font-weight: 600; }
.note { color: var(--muted); font-size: 0.88rem; margin: 10px 0 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 12px;
        margin: 12px 0; }
.card h3 { font-size: 1.02rem; margin: 0; }
.card .line { color: var(--muted); margin: 2px 0 8px; font-size: 0.93rem; }
.flow { width: 100%; height: auto; display: block; }
.flow-narrow { display: none; }
@media (max-width: 620px) { .flow-wide { display: none; } .flow-narrow { display: block; }
  .card { padding: 12px 10px 10px; } .opt { display: none; } th, td { padding: 6px 5px; } }
.tablewrap { overflow-x: auto; }
.res { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 8px; font-size: 0.84rem; }
.chip { border-radius: 6px; padding: 2px 8px; background: var(--chip); white-space: nowrap; }
.ok { background: var(--ok-bg); color: var(--ok); } .no { background: var(--no-bg); color: var(--no); }
.meta { color: var(--muted); margin-left: auto; white-space: nowrap; }
.na { color: var(--muted); font-size: 0.88rem; margin: 8px 0 0; }
details { margin-top: 8px; font-size: 0.88rem; }
summary { cursor: pointer; color: var(--accent); }
details ul { margin: 6px 0; padding-left: 18px; } details li { margin: 2px 0; }
.ans { white-space: pre-wrap; background: var(--bg); border-radius: 8px; padding: 8px 10px; margin: 6px 0 0; }
code { font-size: 0.85em; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; background: var(--card); }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }
th { color: var(--muted); font-weight: 600; } td.c, th.c { text-align: center; }
.foot { color: var(--muted); font-size: 0.84rem; margin-top: 18px; }
/* diagram */
.lane { font-size: 10px; letter-spacing: 0.08em; fill: var(--muted); font-weight: 600; }
.lane-note { font-size: 11.5px; fill: var(--muted); font-style: italic; }
.n-q { fill: var(--q); stroke: var(--line-strong); stroke-width: 1.2; }
.n-m, .n-a { fill: var(--accent); }
.n-v { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1.5; }
.n-s { fill: var(--card); stroke: var(--line-strong); stroke-width: 1.2; }
.n-d { fill: var(--store); stroke: var(--line-strong); stroke-width: 1.2; }
.n-e { fill: var(--card); stroke: var(--line-strong); stroke-width: 1.2; }
.n-e2 { fill: var(--store); stroke: var(--line-strong); stroke-width: 1; }
.t { font-size: 11.5px; fill: var(--ink); } .t-on { font-size: 11.5px; fill: var(--on-accent); font-weight: 600; }
.ar { stroke: var(--line-strong); stroke-width: 1.4; fill: none; }
.ar-d { stroke: var(--line-strong); stroke-width: 1.3; fill: none; stroke-dasharray: 4 3; }
.ah { fill: var(--line-strong); }
.c-tag { fill: var(--card); stroke: var(--line-strong); stroke-width: 1; } .ct-tag { font-size: 9.5px; fill: var(--muted); }
.c-rep { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1; } .ct-rep { font-size: 9.5px; fill: var(--ink); }
.c-new { fill: var(--accent); } .ct-new { font-size: 9.5px; fill: var(--on-accent); font-weight: 700; }
"""


def esc(s):
    return html.escape("" if s is None else str(s))


def year_of(doc_id):
    m = re.search(r"(1[34]\d\d)", doc_id or "")
    return m.group(1) if m else (doc_id or "")[:18]


def snip(text, n=110):
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")


def find_record(variant, dirs):
    for d in dirs:
        for p in sorted((R / d).glob("runs_*.jsonl")):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                if r.get("question_id") == QID and r.get("approach") == variant and not r.get("error"):
                    return r, d
    return None, None


def what_it_did(variant, r):
    """Bullets of what the method actually did on this question, read from its record."""
    tr = r.get("trace") or {}
    items = r.get("retrieved_units") or []
    out = []
    if variant == "B0":
        out.append("No evidence at all: the answer comes from the model's memory.")
    elif variant == "B1":
        n_gold = tr.get("gold_units")
        out.append("Given %d gold passages (the ones a human marked)." % (
            n_gold if isinstance(n_gold, int) else len(n_gold or items)))
        for it in items[:2]:
            out.append("<span class='fa' dir='rtl'>%s: «%s»</span>" % (esc(year_of(it.get("doc_id"))),
                                                                     esc(snip(it.get("text")))))
    elif variant in ("B2", "B3"):
        out.append("Ranked %s chunks in scope; packed the top ones." % esc(tr.get("chunks_in_scope", "?")))
        for it in items[:3]:
            out.append("report %s, score %s: <span class='fa' dir='rtl'>«%s»</span>" % (
                esc(year_of(it.get("doc_id"))), esc(round(it.get("score") or 0, 3)), esc(snip(it.get("text"), 95))))
    elif variant == "B4":
        for st in (tr.get("steps") or [])[:5]:
            if st.get("kind") == "route":
                opts = {str(o[0]): o[1] for o in (st.get("options") or []) if isinstance(o, (list, tuple)) and len(o) > 1}
                years = [year_of(opts.get(str(p), "")) for p in (st.get("pick") or [])]
                out.append("call %s: chose the reports of %s." % (esc(st.get("call")), esc(", ".join(y for y in years if y))))
            else:
                n_open = len((st.get("decision") or {}).get("open") or [])
                out.append("call %s: inside <span class='fa' dir='rtl'>«%s»</span> opened %d part(s)." % (
                    esc(st.get("call")), esc(snip(st.get("path") or "", 70)), n_open))
        out.append("Stopped: %s." % esc(tr.get("stop_reason")))
    elif variant == "B5":
        for st in (tr.get("steps") or [])[:6]:
            a = st.get("args") or {}
            out.append("step %s: <code>%s</code> <span dir='auto'>%s</span>" % (
                esc(st.get("step")), esc(st.get("action")), esc(snip(a.get("pattern") or a.get("section") or "", 80))))
        out.append("Stopped: %s after %s agent steps." % (esc(tr.get("stop_reason")), esc(tr.get("agent_steps_used"))))
    elif variant.startswith("OURS"):
        pats = (tr.get("plan") or {}).get("patterns") or []
        out.append("Patterns: <code dir='auto'>%s</code>" % esc(", ".join(pats[:10])))
        for it in items[:3]:
            out.append("report %s (%s): <span class='fa' dir='rtl'>«%s»</span>" % (
                esc(year_of(it.get("doc_id"))), esc(it.get("kind")), esc(snip(it.get("text"), 95))))
    elif variant.startswith("LEX"):
        st = tr.get("stages") or {}
        kw = st.get("keywords") or {}
        c2 = st.get("call2") or {}
        se = st.get("search") or {}
        out.append("Keywords: <code dir='auto'>%s</code>" % esc(", ".join(kw.get("keywords") or [])))
        if c2.get("picked") is not None:
            out.append("From %d phrases found in the reports it picked: <span class='fa' dir='auto'>%s</span>%s" % (
                len(c2.get("vocab") or []), esc(" · ".join(c2.get("picked") or [])) or "none",
                (" and added <span class='fa' dir='auto'>%s</span>" % esc(" · ".join(c2["added"]))) if c2.get("added")
                else ""))
        out.append("The regexes hit %s paragraphs or table rows; %s windows went to the answer." % (
            esc(se.get("hit_units")), esc(len(se.get("selected") or []))))
        for it in items[:2]:
            out.append("report %s: <span class='fa' dir='rtl'>«%s»</span>" % (
                esc(year_of(it.get("doc_id"))), esc(snip(it.get("text"), 95))))
    else:
        st = tr.get("stages") or {}
        plan = st.get("plan") or {}
        terms = (plan.get("terms") or {}).get("fa") or []
        out.append("Plan: kind <b>%s</b>, years %s, terms <span class='fa' dir='rtl'>%s</span>" % (
            esc(plan.get("kind")), esc(plan.get("period")), esc("، ".join(terms[:6]))))
        s = tr.get("stats") or {}
        if variant == "AGENTIC4":
            gg = (st.get("gap_fill") or {}).get("graph")
            if gg and gg.get("missing"):
                picked = "; ".join("%s: %s = %s" % (year_of(p.get("doc_id")), p.get("label") or "", p.get("value") or "")
                                   for p in (gg.get("picked") or [])[:5])
                out.append("Years still missing after the check: %s. The graph offered %s passage(s) from those "
                           "years' reports%s." % (esc(", ".join(str(y) for y in gg["missing"])),
                                                  esc(gg.get("candidates", 0)),
                                                  (": <span class='snip'>%s</span>" % esc(picked)) if picked else ""))
                if (st.get("gap_fill") or {}).get("second_round") == "verify_gap":
                    out.append("The held-back check call looked at them: %s value(s) accepted in that round." %
                               esc((st.get("gap_fill") or {}).get("accepted", 0)))
            else:
                out.append("No year was missing after the check, so the graph had nothing to add.")
            if s.get("graph_quals_applied"):
                out.append("The graph gave %s checked value(s) their definition." % esc(s["graph_quals_applied"]))
        v = st.get("verify") or {}
        batches = v.get("batches") if isinstance(v, dict) else None
        if batches:
            acc = sum(b.get("accepted", 0) for b in batches)
            rej = sum(b.get("model_rejected", 0) + b.get("guard_rejected", 0) for b in batches)
            out.append("Check: the model compared each number with its quote - %d kept, %d thrown out." % (acc, rej))
        series = ((r.get("structured") or {}).get("series") or [])
        if series:
            rows = ", ".join("%s: %s" % (esc(x.get("period")), esc(x.get("value"))) for x in series[:6])
            out.append("Table handed to the answer: <span dir='auto'>%s</span>." % rows)
        comp = (r.get("structured") or {}).get("comparable")
        if comp is not None:
            out.append("Line-up: the definitions are %s." % ("the same" if comp else "<b>different</b> - no trend"))
    return out


# ---- SVG ---------------------------------------------------------------------------------------------------------

def _chip_w(text):
    return 11 + 5.6 * len(text)


def _text(lines, cx, cy, cls):
    y0 = cy - (len(lines) - 1) * LH / 2 + 4
    spans = "".join("<tspan x='%.1f' y='%.1f'>%s</tspan>" % (cx, y0 + i * LH, esc(t)) for i, t in enumerate(lines))
    return "<text class='%s' text-anchor='middle'>%s</text>" % (cls, spans)


def box(nd, x, y, w):
    """One box, its text and its corner chips."""
    k, out = nd["k"], []
    cx, cy = x + w / 2, y + NH / 2
    if k in ("q", "a"):
        out.append("<rect class='n-%s' x='%.1f' y='%.1f' width='%.1f' height='%d' rx='%d'/>" % (k, x, y, w, NH, NH // 2))
    elif k == "e":
        out.append("<rect class='n-e2' x='%.1f' y='%.1f' width='%.1f' height='%d' rx='4'/>" % (x + 6, y - 5, w - 6, NH))
        out.append("<rect class='n-e' x='%.1f' y='%.1f' width='%.1f' height='%d' rx='4'/>" % (x, y, w - 6, NH))
        cx = x + (w - 6) / 2
    elif k == "d":
        e = 7
        out.append("<path class='n-d' d='M%.1f,%.1f A%.1f,%d 0 0 0 %.1f,%.1f V%.1f A%.1f,%d 0 0 1 %.1f,%.1f Z'/>" % (
            x, y + e, w / 2, e, x + w, y + e, y + NH - e, w / 2, e, x, y + NH - e))
        out.append("<ellipse class='n-d' cx='%.1f' cy='%.1f' rx='%.1f' ry='%d'/>" % (cx, y + e, w / 2, e))
        cy += e / 2
    else:
        out.append("<rect class='n-%s' x='%.1f' y='%.1f' width='%.1f' height='%d' rx='9'/>" % (k, x, y, w, NH))
    out.append(_text(nd["t"].split("\n"), cx, cy, "t-on" if k in ("m", "a") else "t"))
    if nd.get("tag"):
        tw = _chip_w(nd["tag"])
        out.append("<rect class='c-tag' x='%.1f' y='%.1f' width='%.1f' height='15' rx='7.5'/>" % (x + 6, y - 8, tw))
        out.append("<text class='ct-tag' x='%.1f' y='%.1f' text-anchor='middle'>%s</text>" % (x + 6 + tw / 2, y + 3,
                                                                                          esc(nd["tag"])))
    right = nd.get("rep") or ("new" if nd.get("new") else None)
    if right:
        cls = "new" if (nd.get("new") and not nd.get("rep")) else "rep"
        tw = _chip_w(right)
        xr = x + w - 6 - tw - (6 if k == "e" else 0)
        out.append("<rect class='c-%s' x='%.1f' y='%.1f' width='%.1f' height='15' rx='7.5'/>" % (cls, xr, y - 8, tw))
        out.append("<text class='ct-%s' x='%.1f' y='%.1f' text-anchor='middle'>%s</text>" % (cls, xr + tw / 2, y + 3,
                                                                                        esc(right)))
    return "".join(out)


def _arrow(x1, y1, x2, y2, uid):
    return "<line class='ar' x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' marker-end='url(#ah-%s)'/>" % (x1, y1, x2, y2, uid)


def flow_svg(m, uid, geo):
    """The two lanes of one method as an SVG, laid out with `geo` (WIDE or NARROW)."""
    width, nw, gap, cols, row_gap = geo["width"], geo["nw"], geo["gap"], geo["cols"], geo["row_gap"]
    prep, run = m["prep"], m["run"]
    body = ["<text class='lane' x='4' y='11'>BEFORE ANY QUESTION · ONCE</text>"]
    if prep:
        py = 26
        pxs = [4 + i * (nw + gap) for i in range(len(prep))]
        for i, nd in enumerate(prep):
            body.append(box(nd, pxs[i], py, nw))
            if i:
                body.append(_arrow(pxs[i - 1] + nw + 1, py + NH / 2, pxs[i] - 3, py + NH / 2, uid))
        prep_bottom = py + NH
        qlabel = prep_bottom + 30
    else:
        body.append("<text class='lane-note' x='4' y='30'>%s</text>" % esc(
            m.get("prep_note") or "Nothing is prepared in advance."))
        qlabel = 60
    qtop = qlabel + 16
    body.append("<text class='lane' x='4' y='%d'>FOR EACH QUESTION</text>" % qlabel)
    pos = []
    for i in range(len(run)):
        r, c = divmod(i, cols)
        if r % 2:
            c = cols - 1 - c                      # the phone layout snakes back, so every arrow is short
        pos.append((4 + c * (nw + gap), qtop + r * (NH + row_gap)))
    for i, nd in enumerate(run):
        x, y = pos[i]
        body.append(box(nd, x, y, nw))
        if i:
            (x0, y0) = pos[i - 1]
            if y0 == y:
                if x > x0:
                    body.append(_arrow(x0 + nw + 1, y + NH / 2, x - 3, y + NH / 2, uid))
                else:
                    body.append(_arrow(x0 - 1, y + NH / 2, x + nw + 3, y + NH / 2, uid))
            else:
                body.append(_arrow(x0 + nw / 2, y0 + NH + 1, x + nw / 2, y - 3, uid))
    if prep and m.get("uses") is not None and prep[-1]["k"] == "d":
        sx = 4 + (len(prep) - 1) * (nw + gap) + nw / 2
        tx, ty = pos[m["uses"]][0] + nw / 2, pos[m["uses"]][1]
        if run[m["uses"]]["k"] == "e":
            tx -= 3
        mid = prep_bottom + 12
        body.append("<path class='ar-d' d='M%.1f,%.1f V%.1f H%.1f V%.1f' marker-end='url(#ah-%s)'/>" % (
            sx, prep_bottom + 1, mid, tx, ty - 3, uid))
    rows = (len(run) - 1) // cols + 1
    height = qtop + rows * NH + (rows - 1) * row_gap + 6
    label = " → ".join(nd["t"].replace("\n", " ") for nd in run)
    defs = ("<defs><marker id='ah-%s' viewBox='0 0 8 8' refX='7' refY='4' markerWidth='7' markerHeight='7' "
            "orient='auto-start-reverse'><path class='ah' d='M0,0 L8,4 L0,8 z'/></marker></defs>" % uid)
    return ("<svg class='flow' viewBox='0 -2 %d %d' role='img' aria-label='%s'><title>%s</title>%s%s</svg>" % (
        width, height + 2, esc(label), esc(label), defs, "".join(body)))


def legend():
    def sw(inner):
        return "<svg viewBox='0 0 40 26' aria-hidden='true'>%s</svg>" % inner
    items = [
        (sw("<rect class='n-q' x='2' y='4' width='36' height='18' rx='9'/>"), "Question"),
        (sw("<rect class='n-m' x='2' y='4' width='36' height='18' rx='5'/>"), "Model call (the 4B model)"),
        (sw("<rect class='n-v' x='2' y='4' width='36' height='18' rx='5'/>"), "Small vector model"),
        (sw("<rect class='n-s' x='2' y='4' width='36' height='18' rx='5'/>"), "Plain code: search, regex, SQL"),
        (sw("<path class='n-d' d='M4,8 A16,4 0 0 0 36,8 V20 A16,4 0 0 1 4,20 Z'/>"
            "<ellipse class='n-d' cx='20' cy='8' rx='16' ry='4'/>"), "Stored before any question"),
        (sw("<rect class='n-e2' x='7' y='2' width='30' height='20' rx='3'/>"
            "<rect class='n-e' x='3' y='5' width='30' height='20' rx='3'/>"), "Evidence the model reads (≤ 6,000 chars)"),
    ]
    return "<div class='legend'>%s</div>" % "".join("<span>%s%s</span>" % (s, esc(t)) for s, t in items)


def score(q, r):
    """-> (result chips html, scorer output, wrong values), from the study's timeline scorer."""
    s = TL.score_series(q, r, scoring="text")
    hits = s.get("tl_period_hits") or {}
    out = ["<span class='chip %s'>%s %s</span>" % ("ok" if ok else "no", esc(p), "✓" if ok else "✗")
           for p, ok in hits.items()]
    if s.get("tl_false_trend"):
        out.append("<span class='chip no'>calls it a trend ✗</span>")
    elif s.get("tl_comparability_flag"):
        out.append("<span class='chip ok'>says the years can't be compared ✓</span>")
    else:
        out.append("<span class='chip'>no trend claimed</span>")
    wrong = (s.get("tl_n_pred_periods") or 0) - sum(1 for v in hits.values() if v)
    return out, s, max(0, wrong)


def build():
    q = next(x for x in load_questions(QFILE) if x.id == QID)
    series = q.meta["answer_series"]
    head = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'><title>How Each Method Works</title>"
            "<link rel='preconnect' href='https://fonts.googleapis.com'>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap'>"
            "<style>%s</style></head><body><main>" % CSS)
    parts = [head, "<h1>How each method works</h1>",
             "<p class='sub'>Every method gets the same question and the same small model (gemma, 4B). What differs "
             "is how it finds the text the model reads. Each diagram shows what is prepared once, before any "
             "question, and what happens for each question. <b>Filled boxes are model calls.</b></p>", legend()]
    parts.append("<div class='example'><p class='q-fa fa' dir='rtl'>%s</p><p class='q-en'>Example question: %s</p>"
                 % (esc(q.question_fa), esc(q.question_en)))
    parts.append("<div class='strip'><span class='chip'>Right answer</span>%s</div>" % "".join(
        "<span class='yr'>%s · <b>%s</b></span>" % (esc(s["period"]), "{:,}".format(int(s["value"]))) for s in series))
    parts.append("<p class='note'>Each year's report counts people differently (students and graduates, then all "
                 "staff, then several companies), so a right answer gives the four numbers and does <b>not</b> call "
                 "it growth. Under each method below: what it got right on this question.</p></div>")
    summary = []
    for gi, (gname, gline) in enumerate(GROUPS):
        parts.append("<h2>%s</h2><p class='gline'>%s</p>" % (esc(gname), esc(gline)))
        for m in (x for x in METHODS if x["g"] == gi):
            uid = re.sub(r"[^A-Za-z0-9]", "", m["v"])
            parts.append("<section class='card'><h3>%s</h3><p class='line'>%s</p>" % (esc(m["name"]), esc(m["line"])))
            parts.append("<div class='flow-wide'>%s</div>" % flow_svg(m, uid + "w", WIDE))
            parts.append("<div class='flow-narrow'>%s</div>" % flow_svg(m, uid + "n", NARROW))
            r, _d = find_record(m["v"], m["dirs"])
            if r is None:
                why = ("Not run on this question: preparing its reports takes hours of model time."
                       if m["v"] == "OURS" else "Not run on this question yet.")
                parts.append("<p class='na'>%s</p></section>" % esc(why))
                summary.append((m["name"], None))
                continue
            cs, s, wrong = score(q, r)
            secs = (r.get("latency_ms") or 0) / 1000
            parts.append("<div class='res'>%s<span class='meta'>%s model call%s · %.1f s</span></div>" % (
                "".join(cs), r.get("llm_calls"), "" if r.get("llm_calls") == 1 else "s", secs))
            parts.append("<details><summary>What it did on this question</summary><ul>%s</ul>"
                         "<div class='ans fa' dir='auto'>%s</div></details></section>" % (
                             "".join("<li>%s</li>" % b for b in what_it_did(m["v"], r)), esc(r.get("answer_text"))))
            summary.append((m["name"], (s, wrong, r)))
    parts.append("<h2>On this one question</h2><div class='tablewrap'><table><tr><th>Method</th>"
                 "<th class='c'>Years right</th><th class='c opt'>Wrong numbers</th><th>Trend</th>"
                 "<th class='c'>Model calls</th><th class='c opt'>Seconds</th></tr>")
    for name, v in summary:
        if v is None:
            parts.append("<tr><td>%s</td><td class='c' colspan='5'>not run</td></tr>" % esc(name))
            continue
        s, wrong, r = v
        right = sum(1 for x in (s.get("tl_period_hits") or {}).values() if x)
        trend = ("<span class='chip no'>trend ✗</span>" if s.get("tl_false_trend") else
                 "<span class='chip ok'>not comparable ✓</span>" if s.get("tl_comparability_flag") else "—")
        parts.append("<tr><td>%s</td><td class='c'>%d of 4</td><td class='c opt'>%d</td><td>%s</td>"
                     "<td class='c'>%s</td><td class='c opt'>%.1f</td></tr>" % (
                         esc(name), right, wrong, trend, r.get("llm_calls"), (r.get("latency_ms") or 0) / 1000))
    parts.append("</table></div><p class='foot'>A dev question, not the test set. One question is an illustration, not a "
                 "result: the comparisons in the paper are over the full question sets.</p></main></body></html>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    return OUT, summary


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    out, summary = build()
    print("wrote %s (%d methods, %d with a record)" % (out, len(summary), sum(1 for _l, v in summary if v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
