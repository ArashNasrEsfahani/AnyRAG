"""LEX5b approach (work/datasets_v2/lex5b/PROGRESS.md; the configuration admitted by the declared offline rule, S-c):

    1. question encoded once by bge-m3 (dense + sparse; inside the timed query); BM25 + dense + sparse top 40 over the
       pool's chunks; F2 MaxMean fusion over the REAL channel scores (lex8 fusion) -> top 40 chunks
    2. immutable lex4 intact packages; the F2-intact fallback first (F2 packages in order, exact/contained dedupe,
       exact rendered cost, 6,000 chars)
    3. the top 24 shown to the selector through a budgeted view (lex5b/views.py): doc legend, '[n] D<k> > section',
       one excerpt cap per question fitted to a 2,600-token prompt estimate; view 'literal' (query-word window) is the
       admitted default, 'smart' (best lex8 units by R_unit) and 'full' are the offline alternatives
    4. ONE selector call (glm-5.3-flash by default): pinned endpoint via OPENROUTER_PROVIDER_PINS, 1 attempt, 3 s total
       deadline with cancellation, 600-token completion cap incl. reasoning, reasoning effort low
    5. strict lex5 bundle grammar (bundles.parse), no repair; any failure -> the fallback
    6. exact bundle packing, then fill (remaining complete bundles, then F2 packages)
    7. the shared answer call on the rendered evidence
trace['lex5b'] records every stage's wall time and the selector's status / reason / usage / provider / raw reply.
"""
import time
from types import MappingProxyType

import numpy as np

from ...normalize import normalize
from ..lex5 import bundles as B
from ..lex5.approach import SELECTORS
from ..lex8 import fusion as FU
from ..lex8 import retrieve as RV
from ..lex8.approach import Lex8
from . import views as VW

LEX5B_VERSION = "lex5b-v1"
LEX5B_DEFAULTS = {"depth": 40, "top": 24, "view": "literal", "selector": "glm", "deadline_s": 3.0,
                  "max_tokens_total": 600, "reasoning_effort": "low", "attempts": 1, "budget": 6000,
                  "prompt_tokens": VW.PROMPT_TOKENS}


class Lex5b(Lex8):
    name = "LEX5B"
    index_neutral_flags = ()

    def config_signature(self):
        from ...retrieval import hybrid as H
        return {"hybrid": H.signature(self.cfg), "lex5b": LEX5B_DEFAULTS, "version": LEX5B_VERSION,
                "views": VW.VIEWS_VERSION, "bundles": B.BUNDLES_VERSION, "fusion": FU.FUSION_VERSION}

    def _selector(self):
        if getattr(self, "_sel", None) is None:
            from ...llm_openrouter import OpenRouterReader, model_info
            mid = SELECTORS[LEX5B_DEFAULTS["selector"]]
            info = model_info(mid, self.cfg.project_root / "work" / "openrouter" / "catalog.json")
            # optional selector-only pin (dev60 provider probe): env LEX5B_SELECTOR_PIN = JSON {"provider": tag,
            # "params": [...]} for the SAME model id; unset -> OPENROUTER_PROVIDER_PINS as before (the answer reader
            # is never affected)
            import json as _json
            import os as _os
            pin = _json.loads(_os.environ["LEX5B_SELECTOR_PIN"]) if _os.environ.get("LEX5B_SELECTOR_PIN") else None
            self._sel = OpenRouterReader(self.cfg, self.client.ledger, mid, supported=info.get("params"), provider=pin,
                                         attempts=LEX5B_DEFAULTS["attempts"], deadline_s=LEX5B_DEFAULTS["deadline_s"],
                                         max_tokens_total=LEX5B_DEFAULTS["max_tokens_total"],
                                         reasoning_effort=LEX5B_DEFAULTS["reasoning_effort"])
        return self._sel

    def _prep(self, corpus_set, hx):
        six = super()._prep(corpus_set, hx)
        VW.attach_starts(six)
        from ..lex4 import core as K
        self._P = K.Packager(hx, corpus_set)
        try:                                   # untimed: open the deadline path's keep-alive connection (free GET)
            self._selector().warm_deadline()
        except Exception:  # noqa: BLE001
            pass
        return six

    def _answer(self, query, index):
        from ...llm import BudgetExceeded
        T0 = time.perf_counter()
        six, P = self._six, self._P
        budget = int(getattr(self, "budget_chars", None) or LEX5B_DEFAULTS["budget"])   # budget_matrix cells override
        st = {"version": LEX5B_VERSION, "device": getattr(Lex8._enc, "device", None), "budget": budget}
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        t = time.perf_counter()
        q = Lex8._enc.encode([normalize(query.text)], max_length=128, colbert=False)[0]
        st["t_encode_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        # 1. F2 MaxMean over the chunk channels
        t = time.perf_counter()
        mask, _ = six.hx.scope_mask(doc_ids)
        arrays, valid = RV.chunk_channels(six, query.text, q, mask)
        lists = {ch: FU.top_ids(arr, valid[ch], LEX5B_DEFAULTS["depth"]) for ch, arr in arrays.items()}
        order, sc, _N = FU.fuse(arrays, lists, "maxmean")
        order = [int(c) for c in order[:LEX5B_DEFAULTS["depth"]]]
        Ru = None
        view = LEX5B_DEFAULTS["view"]
        if view == "smart":
            umask = six.unit_mask(doc_ids)
            ua, uv = RV.unit_channels(six, query.text, q, umask)
            ul = {ch: FU.top_ids(arr, uv[ch], LEX5B_DEFAULTS["depth"]) for ch, arr in ua.items()}
            _o, Ru, _Nu = RV.fuse_all(ua, ul, six.nu)
        st["t_f2_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        # 2. packages + fallback
        t = time.perf_counter()
        pk = {c: MappingProxyType(dict(P.package(c))) for c in order}
        f2_pks = [pk[c] for c in order]
        rank = {c: i + 1 for i, c in enumerate(order)}
        cost = B.Coster(P.cost)
        fallback = B.hybrid_intact(f2_pks, cost, budget)
        st["t_package_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        # 3. view
        t = time.perf_counter()
        present = f2_pks[:LEX5B_DEFAULTS["top"]]
        v = VW.Viewer(six, index.corpus, query.text, present, Ru if Ru is not None else np.zeros(six.nu))
        prompt, meta, cap, est = v.fit(view)
        st["t_view_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        sel = {"model": None, "status": None, "reason": None, "prompt_chars": len(prompt), "shown": len(present),
               "view": view, "cap": cap, "est_tokens": round(est)}
        client = self._selector()
        sel["model"], sel["provider_pin"] = client.model_id, client.provider
        # 4. one selector call
        t = time.perf_counter()
        text, finish = None, None
        try:
            res = client.chat([{"role": "user", "content": prompt}], purpose="lex5b_select",
                              num_predict=LEX5B_DEFAULTS["max_tokens_total"])
            text, finish = res.text, res.done_reason
            sel.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                       provider=getattr(res, "provider", None), finish=finish, usage_known=True, raw=(text or "")[:2000])
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001  deadline, HTTP error -> fallback, no repair
            sel.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                       error=str(e)[:300], usage_known=False)
        sel["t_select_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        # 5-6. strict validation, exact packing, fill
        t = time.perf_counter()
        shown = fallback
        if text is not None:
            pr = B.parse(text, len(present), finish_reason=finish)
            sel.update(status=pr["status"], reason=pr["reason"], n_parts=pr["n_parts"],
                       parts=[[list(b) for b in alts] for alts in pr["parts"]])
            if pr["ok"]:
                parts_pk = [[tuple(present[i - 1] for i in b) for b in alts] for alts in pr["parts"]]
                shown, info = B.pack_exact(parts_pk, rank, cost, budget)
                shown, finfo = B.fill(shown, parts_pk, f2_pks, cost, budget, bundles=True, rrf=True)
                sel.update(pack=info, fill=finfo,
                           bundles=[[[present[i - 1]["chunk"] for i in b] for b in alts] for alts in pr["parts"]])
        sel["fallback"] = sel["status"] == "fail"
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st["selector"] = sel
        # 7. evidence + the shared answer call
        items = [P.item(p) for p in shown]
        ev = self.pack(items, index)
        st.update(evidence_items=ev.n_included, evidence_chars=ev.chars, cut=ev.last_item_cut,
                  exact_cost=cost(shown), cost_matches=cost(shown) == ev.chars,
                  shown_chunks=[p["chunk"] for p in shown], fallback_chunks=[p["chunk"] for p in fallback],
                  f2_top40=order, f2_scores={str(c): round(float(sc[c]), 5) for c in order},
                  excerpts=[{"chunk": m["chunk"], "chars": m["chars"]} for m in meta])
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        trace = {"lex5b": st, "retrieval_ms": st["t_retrieval_ms"], "evidence_items": ev.n_included,
                 "last_item_cut": ev.last_item_cut, "prompt": prompt}
        return self._result(answer, ev, items, trace)
