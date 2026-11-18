"""Analysis 1, pack-level random effect. All six readers answered byte-identical packs, so answers to the same
(question, arm) pack share pack quality; a question-only random intercept treats the six readers' answers to one pack
as conditionally independent and overstates the information about arm contrasts. This adds a random intercept for
the pack (question x arm), nested in question: Laplace ML (glmm_final.laplace_nll with outer group = question,
inner unit = pack, 6 answers per pack). Writes glmm_pack_results.json.

    .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/glmm_pack.py
"""
import json, os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.stats import chi2
import glmm_final as M
G0 = M.G0
import s5_lib as L


def main():
    t0 = time.time()
    res = {"method": "ML, Laplace approximation; random intercepts for question and for pack (question x arm) nested in question",
           "fits": {}}
    F = res["fits"]
    for scope, qset in (("answerable90", G0.D.ans), ("all96", G0.D.qids)):
        for fill in (0, 1):
            for form in (0, 1, 2):
                X, y, qi, ci, names, nq, nc = G0.build(qset, fill, form)
                arm_idx = np.argmax(X[:, 1:8], axis=1) + 1
                arm_idx[X[:, 1:8].sum(1) == 0] = 0
                pack = qi * 8 + arm_idx
                r = M.fit_laplace(X, y, pack, qi, nq * 8, nq, "cq")
                # fit_laplace's layout for 'cq' returns (beta, log s_outer, log s_inner) = (beta, log s_question, log s_pack)
                p = len(names)
                rec = M.record(r, names, form, "cq", y, nq, nq)
                rec["sigma_pack"] = rec.pop("sigma_question")
                rec["sigma_question"] = rec.pop("sigma_component")
                rec["log_sigma_pack_se"] = rec.pop("log_sigma_question_se")
                rec["log_sigma_question_se"] = rec.pop("log_sigma_component_se", None)
                key = "LAPLACE|%s|unres=%d|form=%d|re=q+pack" % (scope, fill, form)
                F[key] = rec
                c = rec["contrasts"]
                print("%-46s nll=%.4f conv=%s g=%.1e s_q=%.3f s_pack=%.3f LEX-HYB %.3f(%.3f) LEX-HR %.3f(%.3f) (%.0fs)" % (
                    key, rec["nll"], rec["converged"], rec["grad_max"], rec["sigma_question"], rec["sigma_pack"],
                    c["LEX5B_6000 vs HYBRID_6000"]["logOR"], c["LEX5B_6000 vs HYBRID_6000"]["se"],
                    c["LEX5B_6000 vs HR_6000"]["logOR"], c["LEX5B_6000 vs HR_6000"]["se"], time.time() - t0), flush=True)
    base = json.loads((L.OUT / "glmm_final_results.json").read_text())["fits"]
    lrt = {}
    for scope in ("answerable90", "all96"):
        for fill in (0, 1):
            b = "LAPLACE|%s|unres=%d|form=%%d|re=q+pack" % (scope, fill)
            for hi, lo, what in ((1, 0, "pipeline:reader"), (2, 1, "budget:reader + pipeline:budget:reader"), (2, 0, "all reader interactions")):
                st = 2 * (F[b % lo]["nll"] - F[b % hi]["nll"]); df = F[b % hi]["n_fixed"] - F[b % lo]["n_fixed"]
                lrt["%s: form%d vs form%d (%s)" % (b.replace("|form=%d", ""), hi, lo, what)] = dict(stat=st, df=df, p=float(chi2.sf(max(st, 0), df)))
            for form in (0, 1, 2):
                st = 2 * (base["LAPLACE|%s|unres=%d|form=%d|re=q" % (scope, fill, form)]["nll"] - F[b % form]["nll"])
                lrt["LAPLACE|%s|unres=%d|form=%d: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2)" % (scope, fill, form)] = dict(
                    stat=st, df=1, p=float(0.5 * chi2.sf(max(st, 0), 1)))
    res["lrt"] = lrt
    for k, v in lrt.items():
        print("LRT %-100s stat=%.3f df=%d p=%.4g" % (k, v["stat"], v["df"], v["p"]))
    (L.OUT / "glmm_pack_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
