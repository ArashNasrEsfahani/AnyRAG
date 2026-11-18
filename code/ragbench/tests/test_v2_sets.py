"""Datasets-v2 corpus sets BOOK_fa / BOOK_en (letter K) and LAW_fa / LAW_en (letter L), loaded from manifest_v2.csv.

    .venv/Scripts/python.exe -m ragbench.tests.test_v2_sets          # offline; no model call

Offline unit checks run on a temporary manifest_v2 that points at real extractions when they exist; the live checks
load the four real sets and are reported as SKIP when manifest_v2.csv or an extraction is missing.
"""
import copy
import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench.config import load_config  # noqa: E402
from ragbench.corpus import registry as R  # noqa: E402
from ragbench.corpus import extracted  # noqa: E402
from ragbench.contracts import Question  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail else ""), flush=True)
    return cond


def skip(name, why):
    print("SKIP %s - %s" % (name, why), flush=True)


def test_names():
    for n in ("BOOK_fa", "BOOK_en", "LAW_fa", "LAW_en"):
        check("set_names.%s" % n, n in R.SET_NAMES)
    check("set_names.old_kept", R.SET_NAMES[:10] == ("A", "A_fa", "A_en", "B", "B_fa", "B_en", "C", "C_fa", "D", "D_en"))
    check("letters", R.CORPUS_LETTERS == ("A", "B", "C", "D", "K", "L"), R.CORPUS_LETTERS)
    check("set_name_for.K", R.set_name_for("K", "fa") == "BOOK_fa" and R.set_name_for("K", "en") == "BOOK_en")
    check("set_name_for.L", R.set_name_for("L", "fa") == "LAW_fa" and R.set_name_for("L", "en") == "LAW_en")
    check("set_name_for.domain", R.set_name_for("BOOK", "en") == "BOOK_en" and R.set_name_for("LAW", "fa") == "LAW_fa")
    check("set_name_for.old", R.set_name_for("C", "fa") == "C" and R.set_name_for("D", "en") == "D_en"
          and R.set_name_for("A", "fa") == "A_fa")
    try:
        R.set_name_for("K", "de")
        check("set_name_for.bad_lang", False)
    except ValueError:
        check("set_name_for.bad_lang", True)
    check("corpus_letter", R.CorpusSet("BOOK_fa", []).corpus == "K" and R.CorpusSet("LAW_en", []).corpus == "L"
          and R.CorpusSet("C", []).corpus == "C" and R.CorpusSet("A_en", []).corpus == "A")
    check("v2_key", R.v2_key(r"law_en\eurlex-32016R0679_gdpr_en.html", "en") == "eurlex-32016R0679_gdpr"
          and R.v2_key("book_fa/wikisource-x_fa.html", "fa") == "wikisource-x" and R.v2_key("a/b_c.html", "en") == "b_c")
    try:
        R.load_set("BOOK")
        check("load_set.bare_domain_refused", False)
    except ValueError:
        check("load_set.bare_domain_refused", True)


def test_question():
    base = dict(id="t1", question_fa="?", question_en="?", answer_fa="x", answer_en="x", answer_type="short",
                gold_units={"en": ["L.x.en::a#b"]}, doc_scope="corpus", type="lookup")
    check("question.K_valid", "corpus" not in Question(corpus="K", **base).validate())
    check("question.L_valid", "corpus" not in Question(corpus="L", **base).validate())
    check("question.X_invalid", "corpus" in Question(corpus="X", **base).validate())
    q = Question(corpus="L", **base)
    check("question.supports_en", q.supports("en-en") and q.supports("fa-en"))
    check("question.no_fa_gold", not q.supports("fa-fa"))


def test_pools():
    from ragbench.pools import corpus_set_names
    check("pools.set_names", corpus_set_names("K") == ("BOOK_en", "BOOK_fa") and corpus_set_names("L") == ("LAW_en", "LAW_fa"))


def _usable_rows(cfg, lane):
    root = cfg.path("extracted_root")
    for r in R.read_manifest_v2(cfg):
        if r.get("lane") == lane and extracted.is_usable(extracted.out_dir_for(root, r["local_path"]))[0]:
            yield r


def test_loader_on_temp_manifest(cfg):
    real = next(iter(_usable_rows(cfg, "LAW_en")), None)
    if real is None:
        skip("loader.temp_manifest", "no usable LAW_en extraction yet")
        return
    with tempfile.TemporaryDirectory() as td:
        man = Path(td) / "manifest_v2.csv"
        cols = list(real.keys())
        rows = [dict(real),
                dict(real, local_path=r"law_en\does-not-exist_en.html"),                # no extraction -> skipped
                dict(real, lane="LAW_fa"),                                               # other lane -> ignored here
                dict(real, lane="LAW_en", language="fa", local_path=r"law_en\y_fa.html"),  # language clash -> skipped
                dict(real)]                                                              # duplicate -> skipped
        with open(man, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
        c2 = copy.deepcopy(cfg)
        c2.data["paths"]["manifest_v2"] = str(man)
        check("loader.manifest_path", R.manifest_v2_path(c2) == man)
        docs, skipped = R._load_v2(c2, "LAW_en")
        want = "L.%s.en" % R.v2_key(real["local_path"], "en")
        check("loader.one_doc", [d.id for d in docs] == [want], [d.id for d in docs])
        check("loader.doc_fields", docs and docs[0].lang == "en" and docs[0].corpus == "L", docs and docs[0].corpus)
        reasons = " | ".join(s["reason"] for s in skipped)
        check("loader.skipped", len(skipped) == 3 and "duplicate" in reasons and "language" in reasons, reasons)
        docs_fa, _ = R._load_v2(c2, "LAW_fa")
        check("loader.lane_language_filter", docs_fa == [], [d.id for d in docs_fa])
    c3 = copy.deepcopy(cfg)
    c3.data["paths"]["manifest_v2"] = str(Path(tempfile.gettempdir()) / "no_such_manifest_v2.csv")
    check("loader.missing_manifest_empty", R._load_v2(c3, "BOOK_en") == ([], []))


def test_live_sets(cfg):
    if not R.manifest_v2_path(cfg).is_file():
        skip("live", "no %s" % R.manifest_v2_path(cfg))
        return
    for name in R.V2_SET_NAMES:
        rows = [r for r in R.read_manifest_v2(cfg) if r.get("lane") == name]
        if not rows:
            skip("live.%s" % name, "no manifest rows")
            continue
        cs = R.load_set(name, cfg)
        letter = R.V2_DOMAINS[name.split("_")[0]]
        ok = len(cs) == len(rows) and all(d.id.startswith(letter + ".") and d.lang == name[-2:] for d in cs)
        check("live.%s" % name, ok and not cs.skipped and all(len(d.sections) > 1 for d in cs),
              "%d docs of %d rows, %d skipped" % (len(cs), len(rows), len(cs.skipped)))


def main():
    cfg = load_config()
    test_names()
    test_question()
    test_pools()
    test_loader_on_temp_manifest(cfg)
    test_live_sets(cfg)
    bad = [k for k, v in RESULTS.items() if not v]
    print("test_v2_sets: %d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)) + (" FAILED: %s" % bad if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
