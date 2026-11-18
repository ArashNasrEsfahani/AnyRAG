"""LEX5 approach: LLM-selected hybrid retrieval (work/analysis/LEX5_DESIGN_FINAL.md, spec 1-7).

    1. question hybrid: BM25 top 40 and bge-m3 dense top 40 over the pool, equal-weight RRF (k=60), existing ties
    2. immutable intact packages (lex4 Packager: verbatim chunk, <= 300 chars sentence completion, 1,200 body ceiling,
       table caption/heading prefixes, provenance of every displayed interval); the HYBRID-intact fallback is built
       FIRST with the same rules (exact / fully-contained dedupe only, exact rendered cost, 6,000 chars)
    3. the top TOP (24) packages shown to the selector: '[n] <title> > <section path>' + the view (full package)
    4. ONE selector call (default openai/gpt-oss-20b): pinned OpenRouter endpoint (no provider fallback), reasoning
       low, 1 attempt, 3 s total deadline with real cancellation, 1,024-token completion cap incl. reasoning
    5. strict validation (bundles.parse); any failure -> the fallback, no repair; all-NONE is valid (and distinct)
    6. exact bundle packing (bundles.pack_exact), then fill (remaining complete bundles, then RRF packages)
    7. the shared answer call (contracts.Approach.generate_answer) on the rendered evidence

Flags (ablations; variant 'LEX5+<flags>'):
    selector=oss20b|glm   selector model (SELECTORS)            view=full|query_hit|head   the selector's view
    top=24|40             candidates shown                      no_fill    neither fill stage
    flat                  flattened packing (no bundles)        greedy     greedy bundle packing (not exact)
    permute               one fixed (seeded) permutation of the shown candidate order
HYBRID-intact is the HYBRID flag 'intact' (approaches/hybrid.py -> answer_intact below): steps 1, 2 and 7 only.
trace['lex5'] records every stage's wall time, the selector's status / reason / usage / provider / cost and the
packing decisions, so the resource table can be rebuilt from records and ledgers alone.
"""
import random
import time
from types import MappingProxyType

from ...retrieval import hybrid as H
from ..hybrid import HybridRetrieval
from ..lex4 import core as K
from . import bundles as B
from . import select as S

LEX5_VERSION = "lex5-v1"
SELECTORS = {"oss20b": "openai/gpt-oss-20b", "glm": "z-ai/glm-5.3-flash"}
LEX5_DEFAULTS = {"depth": 40, "top": 24, "view": "full", "selector": "oss20b", "deadline_s": 3.0,
                 "max_tokens_total": 1024, "reasoning_effort": "low", "attempts": 1, "budget": 6000,
                 "permute_seed": 0}
VIEWS = {"full": "full", "query_hit": "snippet", "head": "head"}

PROMPT = """You are selecting evidence passages for a question. Read the question and the numbered passages.

Question: {question}

Passages:
{passages}

Task:
1. Split the question into the separate parts it asks for: one part per separate piece of information needed to answer it. A simple question has one part. Use at most 8 parts.
2. For each part, give the passages that together fully support its answer. Include every passage the answer depends on: each value or operand, each condition, qualifier, exception or definition - not only the passage that states the final answer. Join passages that are needed together with +. If a different set of passages would also fully support the part on its own, give it after | as an alternative (at most 2 alternatives, best first). Do not list passages that are only on a related topic. If no passage supports a part, write NONE.

Reply with exactly these lines and nothing else (numbers are passage numbers). The first line gives the number of parts, then one line per part, and the last line is the word END. An alternative must be a different set of passages.
parts=<number of parts>
P1: <passages joined by +> | <alternative passages joined by +>
P2: NONE
END
Always finish your reply with the line END."""


def freeze(pk):
    return MappingProxyType(dict(pk))


class Lex5Select(HybridRetrieval):
    name = "LEX5"
    index_neutral_flags = ("selector", "view", "top", "no_fill", "flat", "greedy", "permute")

    def config_signature(self):
        return {"hybrid": H.signature(self.cfg), "lex5": LEX5_DEFAULTS, "version": LEX5_VERSION,
                "bundles": B.BUNDLES_VERSION, "core": K.CORE_VERSION}

    # -- selector client (always OpenRouter; its calls land in the same ledger scope as the answer call) -----------
    def _selector(self):
        if getattr(self, "_sel", None) is None:
            from ...llm_openrouter import OpenRouterReader, model_info
            mid = SELECTORS[self.flags.get("selector") or LEX5_DEFAULTS["selector"]]
            info = model_info(mid, self.cfg.project_root / "work" / "openrouter" / "catalog.json")
            self._sel = OpenRouterReader(self.cfg, self.client.ledger, mid, supported=info.get("params"),
                                         attempts=LEX5_DEFAULTS["attempts"], deadline_s=LEX5_DEFAULTS["deadline_s"],
                                         max_tokens_total=LEX5_DEFAULTS["max_tokens_total"],
                                         reasoning_effort=LEX5_DEFAULTS["reasoning_effort"])
        return self._sel

    def _warm(self):
        try:                                   # untimed: open the deadline path's keep-alive connection (free GET)
            self._selector().warm_deadline()
        except Exception:  # noqa: BLE001
            pass

    def _build_index(self, corpus_set, index_dir):
        hx = super()._build_index(corpus_set, index_dir)
        self._warm()
        return hx

    def _load_index(self, corpus_set, index_dir):
        hx = super()._load_index(corpus_set, index_dir)
        self._warm()
        return hx

    def _answer(self, query, index):
        return run_lex5(self, query, index, select=True)


# ----------------------------------------------------------------------------------------------------------------
# shared by LEX5 and HYBRID+intact
# ----------------------------------------------------------------------------------------------------------------

def _packager(ap, index):
    cache = ap.__dict__.setdefault("_lex5_packagers", {})
    key = index.corpus.name if hasattr(index.corpus, "name") else id(index.corpus)
    if key not in cache:
        cache[key] = K.Packager(index.payload, index.corpus)
    return cache[key]


def answer_intact(ap, query, index):
    return run_lex5(ap, query, index, select=False)


def run_lex5(ap, query, index, select=True):
    from ...contracts import AnswerResult
    fl = ap.flags or {}
    budget = int(ap.cfg.evidence["max_chars"])
    st = {"version": LEX5_VERSION, "budget": budget}
    T0 = time.perf_counter()
    hx = index.payload
    doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
    depth = int(LEX5_DEFAULTS["depth"])
    # 1. hybrid: BM25 top 40 U dense top 40, RRF (every fused chunk kept: <= 80)
    t = time.perf_counter()
    hits = hx.search(query.text, doc_ids=doc_ids, k=2 * depth, depth=depth, client=ap.client)
    fused = [h.chunk for h in hits]
    st["t_hybrid_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    # 2. immutable packages + the fallback first
    t = time.perf_counter()
    P = _packager(ap, index)
    pk = {c: freeze(P.package(c)) for c in fused}
    rrf_rank = {c: i + 1 for i, c in enumerate(fused)}
    rrf_order = [pk[c] for c in fused]
    cost = B.Coster(P.cost)
    fallback = B.hybrid_intact(rrf_order, cost, budget)
    st["t_package_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    shown, sel = fallback, None
    if select:
        top_n = int(fl.get("top") or LEX5_DEFAULTS["top"])
        view = fl.get("view") or LEX5_DEFAULTS["view"]
        top = fused[:top_n]
        present = list(top)
        if fl.get("permute"):
            random.Random(LEX5_DEFAULTS["permute_seed"]).shuffle(present)
        # 3-4. one selector call
        t = time.perf_counter()
        passages = S.views(P, [pk[c] for c in present], query.text, VIEWS[view])
        msgs = [{"role": "user", "content": PROMPT.format(question=query.text.strip(),
                                                          passages="\n\n".join(passages))}]
        st["t_view_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        sel = {"model": None, "status": None, "reason": None, "prompt_chars": len(msgs[0]["content"]),
               "shown": len(present), "view": view}
        client = ap._selector()
        sel["model"] = client.model_id
        sel["provider_pin"] = client.provider
        t = time.perf_counter()
        from ...llm import BudgetExceeded
        text, finish = None, None
        try:
            res = client.chat(msgs, purpose="lex5_select", num_predict=LEX5_DEFAULTS["max_tokens_total"])
            text, finish = res.text, res.done_reason
            sel.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                       provider=getattr(res, "provider", None), finish=finish, usage_known=True, raw=text[:2000])
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001  deadline, HTTP error, no choices -> fallback, no repair
            sel.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                       error=str(e)[:300], usage_known=False)
        sel["t_select_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        # 5. strict validation
        t = time.perf_counter()
        if text is not None:
            pr = B.parse(text, len(present), finish_reason=finish)
            sel.update(status=pr["status"], reason=pr["reason"], n_parts=pr["n_parts"],
                       parts=[[list(b) for b in alts] for alts in pr["parts"]])
            if pr["ok"]:
                parts_pk = [[tuple(pk[present[i - 1]] for i in b) for b in alts] for alts in pr["parts"]]
                # 6. packing + fill
                if fl.get("flat"):
                    shown, info = B.pack_flat(parts_pk, cost, budget)
                    packer = "flat"
                elif fl.get("greedy"):
                    shown, info = B.pack_greedy(parts_pk, cost, budget)
                    packer = "greedy"
                else:
                    shown, info = B.pack_exact(parts_pk, rrf_rank, cost, budget)
                    packer = "exact"
                sel.update(packer=packer, pack=info, n_selected_shown=len(shown))
                if not fl.get("no_fill"):
                    shown, finfo = B.fill(shown, parts_pk, rrf_order, cost, budget,
                                          bundles=not fl.get("flat"), rrf=True)
                    sel["fill"] = finfo
            else:
                shown = fallback
        sel["fallback"] = sel["status"] == "fail"
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st["selector"] = sel
    # 7. evidence and the shared answer call
    items = [P.item(p) for p in shown]
    ev = ap.pack(items, index)
    st.update(evidence_items=ev.n_included, evidence_chars=ev.chars, cut=ev.last_item_cut,
              exact_cost=cost(shown), cost_matches=cost(shown) == ev.chars,
              shown_chunks=[p["chunk"] for p in shown], fallback_chunks=[p["chunk"] for p in fallback],
              top_rrf=fused[:40])
    st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
    t = time.perf_counter()
    answer = ap.generate_answer(query, ev)
    st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    trace = {"hybrid": hits.diagnostics, "lex5": st, "retrieval_ms": st["t_retrieval_ms"],
             "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
    return AnswerResult(answer_text=answer, evidence=ev, retrieved=items, trace=trace)
