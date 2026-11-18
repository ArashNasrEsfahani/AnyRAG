"""Budget matrix cells (2026-09-26, work/datasets_v2/budget_matrix/PROGRESS.md): the evidence budget is a study
variable with two levels, 6,000 and 15,500 rendered chars, for every arm. These classes are the existing arms with
ONLY the pack budget raised to 15,500 chars; retrieval, ordering, selector / fixer and the shared answer call are
unchanged. Run with --approach-class ragbench.approaches.budget_matrix:<Class> (not in the shared registry).

    HR15500     HYBRID+rerank (bge-reranker-v2-m3 over the top 50 fused, top 40 kept) -> HYBRID's own offer / final
                pack (pack_evidence) at 15,500
    Lex7_15500  LEX7.1 (RRF of BM25 + bge-m3 dense + sparse, MMR, order fixer v2 over the top 12) -> the same offer /
                final pack at 15,500
    Lex5b_15500 LEX5B (F2 top 40, literal view of the top 24, one glm selector call, strict grammar); fallback, exact
                bundle packing and fill all at 15,500, then the rendered pack at 15,500
    Setr15500   SETR (2026-10-01, test re-freeze): the zero-shot SetR call over HYBRID's top 20 (unchanged; the call
                does not depend on the budget; it runs again for this cell) -> selected first, then the rest of the
                top 40 -> HYBRID's offer / final pack at 15,500

The packer is the shared pack_evidence in every cell (the 6,000 arms use it at cfg.evidence.max_chars).
"""
from ..contracts import pack_evidence
from .hybrid import HybridRetrieval
from .lex5b.approach import Lex5b
from .lex7.approach import Lex7
from .setr.approach import SetR

BUDGET = 15500
CELLS_VERSION = "budget-matrix-v1"


class _Wide:
    budget_chars = BUDGET

    def pack(self, items, index, budget=None):
        return pack_evidence(items, index.corpus, budget=budget or self.budget_chars, cfg=self.cfg)

    def config_signature(self):
        sig = dict(super().config_signature())
        sig["budget_matrix"] = {"version": CELLS_VERSION, "budget_chars": self.budget_chars}
        return sig


class HR15500(_Wide, HybridRetrieval):
    name = "HR15500"

    def __init__(self, cfg=None, client=None, flags=None):
        f = dict(flags or {})
        f["rerank"] = True
        super().__init__(cfg=cfg, client=client, flags=f)


class Lex7_15500(_Wide, Lex7):
    name = "LEX7_15500"


class Lex5b_15500(_Wide, Lex5b):
    name = "LEX5B_15500"


class Setr15500(_Wide, SetR):
    name = "SETR_15500"
