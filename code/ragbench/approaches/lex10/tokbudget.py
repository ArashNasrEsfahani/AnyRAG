"""LEX10 pack-size rule (declared 2026-09-26, work/datasets_v2/lex10/PROGRESS.md).

The reader's (GLM) tokenizer is not available offline, so prompt tokens are predicted as k_lang * o200k(prompt), with
k_lang = sum(GLM prompt tokens) / sum(o200k tokens) over saved answer calls whose prompt re-renders exactly, plus the
mean completion of those calls. B* = the largest pack size (step 500) whose predicted mean tokens <= cap.
"""


def fit(calls):
    """calls: dicts with lang, o200k, glm_prompt, glm_completion -> {lang: {"k", "completion"}}."""
    out = {}
    for lg in sorted({c["lang"] for c in calls}):
        cs = [c for c in calls if c["lang"] == lg]
        out[lg] = {"k": sum(c["glm_prompt"] for c in cs) / sum(c["o200k"] for c in cs),
                   "completion": sum(c["glm_completion"] for c in cs) / len(cs)}
    return out


def predict(model, lang, o200k_tokens):
    m = model[lang]
    return m["k"] * o200k_tokens + m["completion"]


def pick_budget(mean_by_size, cap):
    """mean_by_size: {pack chars: predicted mean tokens}. The largest size within cap (None if none)."""
    ok = [b for b, t in mean_by_size.items() if t <= cap]
    return max(ok) if ok else None
