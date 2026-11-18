"""SETR (2026-10-01; work/datasets_v2/setr): a literature baseline after SetR (Lee et al., ACL 2025, arXiv 2507.06838,
"Shifting from Ranking to Set Selection for Retrieval Augmented Generation"; code github.com/LGAI-Research/SetR).

SetR fine-tunes Llama-3.1-8B-Instruct on GPT-4o-distilled outputs of ONE prompt ("Selection IRI" mode,
generate_data.py): the model lists the query's information requirements, finds the passages holding each requirement,
then outputs the selected SET of passages ('### Final Selection: [4] [2]'). This arm is the ZERO-SHOT prompted version
of that prompt (no fine-tuning): the same system message and user prompt, verbatim, with the same passage rendering
('[i] text', newlines collapsed to spaces) over the top 20 candidates of a first-stage list.

    1. candidates: HYBRID's own fused list (BM25 + bge-m3 dense, equal-weight RRF k=60, depth 40; exactly
       HybridRetrieval's hx.search) -> top 40 chunks
    2. the top 20 shown to the selector as full chunks (a fixed safety cap of PASSAGE_CAP chars per passage, applied
       identically to every question; chunks are ~1,000 chars so it rarely binds; the trace counts cut passages)
    3. ONE selector call: glm-5.3-flash via the pinned endpoint (OPENROUTER_PROVIDER_PINS, as the LEX5B selector),
       1 attempt, 8 s total deadline with cancellation (CoT is longer than LEX5B's 3 s budget), 2,000-token
       completion cap incl. reasoning, reasoning effort low
    4. parse: the LAST 'Final Selection' marker; its [n] ids (1..20, first occurrence kept, model order). No marker,
       no valid id, a deadline or a call error -> fallback = plain HYBRID order (no repair, no retry)
    5. pack: the selected chunks first (selection order), then the remaining top-40 candidates in HYBRID order, through
       HYBRID's own offer + chunk packer (intact chunks, 6,000 rendered chars), then the shared answer call.
"""
import re
import time

from ...retrieval import hybrid as H
from ..hybrid import HYBRID_DEFAULTS, HybridRetrieval

SETR_VERSION = "setr-zeroshot-v1"
PASSAGE_CAP = 1500
SETR_DEFAULTS = {"top": 20, "depth": 40, "selector": "z-ai/glm-5.3-flash", "deadline_s": 8.0,
                 "max_tokens_total": 2000, "reasoning_effort": "low", "attempts": 1, "passage_cap": PASSAGE_CAP,
                 "budget": 6000}

# verbatim from SetR generate_data.py (mode 'selection IRI'); only {num}, {question}, {context} are filled
SYSTEM = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
USER = ("I will provide you with {num} passages, each indicated by a numerical identifier []. "
        "Select the passages based on their relevance to the search query: {question}.\n\n"
        "{context}\n\n"
        "Search Query: {question}\n\n"
        "Please follow the steps below:\n"
        "Step 1. Please list up the information requirements to answer the query.\n"
        "Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.\n"
        "Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. "
        "Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', "
        "e.g., ### Final Selection: [4] [2].")

_FINAL = re.compile(r"final\s+selection\s*\**\s*:?\s*\**(.*)", re.I)
_ID = re.compile(r"\[\s*(\d{1,3})\s*\]")


def render_passage(text, cap=PASSAGE_CAP):
    t = re.sub(r"\n+", " ", (text or "").strip())
    if len(t) > cap:
        return t[:cap].rstrip() + " …", True
    return t, False


def build_prompt(question, texts, cap=PASSAGE_CAP):
    parts, cut = [], 0
    for i, t in enumerate(texts):
        body, c = render_passage(t, cap)
        cut += c
        parts.append("[%d] %s" % (i + 1, body))
    user = USER.format(num=len(texts), question=question, context="\n".join(parts))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], cut


def parse_selection(text, n):
    """-> (ids 1-based in model order, status). status: ok | empty_reply | no_marker | empty_selection."""
    if not text:
        return [], "empty_reply"
    found = list(_FINAL.finditer(text))
    if not found:
        return [], "no_marker"
    out = []
    for m in _ID.finditer(found[-1].group(1)):
        k = int(m.group(1))
        if 1 <= k <= n and k not in out:
            out.append(k)
    return out, ("ok" if out else "empty_selection")


def reorder(items, sel_ids, top):
    """selected (1-based into items[:top]) first, then every other candidate in the original order."""
    first = [items[i - 1] for i in sel_ids if i - 1 < min(top, len(items))]
    taken = {id(x) for x in first}
    return first + [x for x in items if id(x) not in taken]


class SetR(HybridRetrieval):
    name = "SETR"
    index_neutral_flags = ()

    def config_signature(self):
        sig = super().config_signature()
        sig["setr"] = {"version": SETR_VERSION, **SETR_DEFAULTS}
        return sig

    def _selector(self):
        if getattr(self, "_sel", None) is None:
            from ...llm_openrouter import OpenRouterReader, model_info
            mid = SETR_DEFAULTS["selector"]
            info = model_info(mid, self.cfg.project_root / "work" / "openrouter" / "catalog.json")
            self._sel = OpenRouterReader(self.cfg, self.client.ledger, mid, supported=info.get("params"),
                                         attempts=SETR_DEFAULTS["attempts"], deadline_s=SETR_DEFAULTS["deadline_s"],
                                         max_tokens_total=SETR_DEFAULTS["max_tokens_total"],
                                         reasoning_effort=SETR_DEFAULTS["reasoning_effort"])
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

    def select(self, query, index):
        """Steps 1-4 (no answer call): -> (ordered items, hits, st). The offline harness calls this too."""
        from ...llm import BudgetExceeded
        hx = index.payload
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        t = time.perf_counter()
        k = max(int(HYBRID_DEFAULTS["candidates"]), int(self.cfg.chunking["top_k"]))
        hits = hx.search(query.text, doc_ids=doc_ids, k=k, depth=int(HYBRID_DEFAULTS["depth"]), client=self.client)
        items = [h.to_item(rank=r) for r, h in enumerate(hits)]
        st = {"version": SETR_VERSION, "t_hybrid_ms": round((time.perf_counter() - t) * 1000.0, 1)}
        top = min(int(SETR_DEFAULTS["top"]), len(items))
        msgs, cut = build_prompt(query.text, [it.text for it in items[:top]])
        sel = {"shown": top, "passages_cut": cut, "prompt_chars": sum(len(m["content"]) for m in msgs),
               "status": None, "reason": None}
        client = self._selector()
        sel["model"], sel["provider_pin"] = client.model_id, client.provider
        t = time.perf_counter()
        text = None
        try:
            res = client.chat(msgs, purpose="setr_select", num_predict=SETR_DEFAULTS["max_tokens_total"])
            text = res.text
            sel.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                       provider=getattr(res, "provider", None), finish=res.done_reason, usage_known=True,
                       raw=(text or "")[:4000])
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001  deadline, HTTP error -> fallback, no repair
            sel.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                       error=str(e)[:300], usage_known=False)
        sel["t_select_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        if text is not None:
            ids, status = parse_selection(text, top)
            sel.update(status="ok" if ids else "fail", reason=None if ids else status, selected=ids,
                       selected_chunks=[int(items[i - 1].meta["chunk"]) for i in ids])
        sel["fallback"] = sel["status"] != "ok"
        ordered = list(items) if sel["fallback"] else reorder(items, sel["selected"], top)
        st["selector"] = sel
        st["order40"] = [int(h.chunk) for h in hits]
        return ordered, hits, st

    def pack_ordered(self, ordered, index, query):
        """Step 5 without the answer call: HYBRID's own offer + chunk packer over the SETR order."""
        offered = self._offer(ordered, index)
        return offered, self.final_pack(offered, index, query)

    def _answer(self, query, index):
        T0 = time.perf_counter()
        ordered, hits, st = self.select(query, index)
        t = time.perf_counter()
        offered, ev = self.pack_ordered(ordered, index, query)
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st.update(evidence_items=ev.n_included, evidence_chars=ev.chars,
                  offered=[int(it.meta["chunk"]) for it in offered])
        trace = {"setr": st, "hybrid": hits.diagnostics, "retrieval_ms": st["t_retrieval_ms"],
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut, "rrf_k": H.RRF_K}
        return self._result(answer, ev, ordered, trace)
