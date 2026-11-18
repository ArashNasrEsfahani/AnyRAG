# S14 "bilingual, saved data": language interaction, replication, tokenizers and judges by language

Post-hoc and exploratory (COMMON.md). **No API calls, no GPU** (`CUDA_VISIBLE_DEVICES=""` in every process). Only saved
verdicts, ledgers and packs were read. Every number below comes from `lang.json` / `lang_tables.md` /
`compact_table.md` (task 1) or `tokjudge.json` / `tokjudge_tables.md` (task 2), all in this folder.

## 1. What was run

Code: `s14lib.py` (loaders, permutation tests, MDE), `t1_lang.py` (task 1), `t3_compact.py` (compact table from
`lang.json`), `t2_tok_judge.py` (task 2). Logs: `t1_lang.log`, `t2.log`.

**Verdict sources (read-only, through the owning streams' own loaders):**
- **Judge 1, GLM r0:** every arm except HYBRID_AGENT through `S8_factorial/analyze.py` `J1[...]`, imported as a module;
  its `main()` is guarded. That covers the frozen arms via `report_test.score`, the S2a FUSION/HYBRID/HR/GOLD cells,
  S3/S6 arms from their judge dirs, and LEX5B_3000 from S8 lexbud.
- **HYBRID_AGENT:** the S9 runs and judge dir, with S9's own rule.
- **Other five readers:** `report_test.score` on the replays, for the 8 frozen arms only.
- **GLM replicates r1/r2:** `S2b_variance/exp1_reader`, with S2b's verdict rule. Where an arm has no replicates, the
  replicate row keeps that arm at r0 ("one side replicated"). For FUSION_6000, the 32 answerable cells whose answer is
  the frozen LEX5B_6000 answer (identical pack) take LEX5B_6000's replicate.
- **Judge 2 (mistral-large-2512):** `S8 analyze.J2`. The 8 frozen arms come from S1; the post-hoc arms are S8's partial
  run, which kept filling while this stream ran. Snapshot at 2026-11-10T20:08, resolved answerable cells: FUSION_6000
  88, ITER_BM25 88, SETR_NOFILL 89, GREP_AGENT 14, GOLD 1. HYBRID_AGENT, HYBRID_3000, LEX5B_3000, FUSION_3000 and
  HYBRID_24000 were never sent to judge 2.
- **Reproduction check:** every per-arm and per-language total reproduces the owning stream's numbers, e.g.
  HYBRID_6000 fa 35..36 / en 27; GREP_AGENT 29 / 39; HYBRID_AGENT 39 / 38; GOLD 43 / 40.

**Task 1.** The 14 contrasts A − B (the brief's list, with "GOLD vs best single-pass" run two ways: HR_6000, the best
frozen arm, and HYBRID_24000, the best of the S2a/S8 sweep at 72/90).
- **Statistics per half** (fa 45, en 45): net, wins/losses, exact sign-p range over unresolved cells, and 95% bootstrap
  CIs over questions and over source-family components. These use `revision_stats.compare`, in S8's ast-extracted copy
  with an exact DP sign range above 12 unresolved cells (10,000 resamples, seed 0).
- **Language × arm permutation test** on d_q = A_q − B_q, with statistic T = mean d (en) − mean d (fa), two-sided:
  - question level: 20,000 random relabellings of language over the 90 questions (seed 0);
  - component level: exact, all 184,756 relabellings of the 20 components, 10 as "en". Components nest within
    language, so this is the honest unit.
  - p is a range over every assignment of unresolved cells (enumeration up to 8 cells, otherwise the 4 language corners).
- **Interaction CI:** a stratified bootstrap 95% CI of T.
- **Replication conditions:** the 5 other readers (both arms frozen), r1 and r2, and judge 2 (a half with more than 6
  unresolved cells is "undetermined"). All of it was repeated without the 26 closed-book-correct questions
  (S7 `contam.json`; fa 39, en 25).
- **MDE:** within-language MDE at n = 45 with S5's exact sign-test method (functions copied).
- **Multiplicity:** Holm over the 28 within-language tests and over the 14 interaction tests.

**Task 2.**
- **(a) Tokens per character:** provider-reported answer-call prompt tokens. GLM comes from the frozen ledgers and the
  five others from the replay ledgers, on the byte-identical packs of the 8 frozen arms × 96 questions: 768 calls per
  reader. For every call the ledger's prompt_chars equals the pack's character count (768/768 per reader).
  - Method: S7's per-language OLS, tokens ~ 1 + evidence chars + system chars, giving a fa/en ratio of slopes.
  - CI: a question-cluster bootstrap (2,000, seed 0).
- **(b) Judges by language:**
  - judge 1 vs judge 2 agreement and kappa per language on the S1 GLM cells;
  - each judge vs the human on the 120 reviewed cells;
  - S7's 66 disagreement labels (`errors.json` a_cells) split by language;
  - language differences tested by question-level permutation (10,000) and a stratified bootstrap (2,000).

## 2. Results

### 2.1 Replication table (`compact_table.md`; full per-contrast tables with CIs in `lang_tables.md`)

Net A − B in answers. Primary = GLM r0, judge 1. In the three replication columns each cell gives Persian / English;
"–" means the condition does not exist.

| contrast | fa /45 (sign p) | en /45 (sign p) | no-contam fa /39 · en /25 | 5 readers (net range) | replicates r1; r2 | judge 2 | interaction T pp; p_q; p_c (worst case) |
|---|---|---|---|---|---|---|---|
| HR vs HYBRID @6000 | +0..+1 (1.0) | +6..+8 (.039-.109) | +1..+2 · +4..+5 | −4..+2 / +0..+7 | +1; +0..+2 / +4; +8 | −2 / +4..+6 | +11..+18; .070-.286; .077-.172 |
| Lex-RAG vs FUSION @6000 | +4..+6 (.031-.219) | +5 (.180) | +3..+5 · +4 | – | +2..+3; +3..+4 / +2..+3; +1 | +4 / +2..+6 | −2..+2; 1.0; .79-1.0 |
| Lex-RAG vs FUSION @3000 | +7..+8 (.039-.065) | +8 (.008) | +7..+8 · +6 | – | – | – | 0..+2; 1.0; .86-1.0 |
| Lex-RAG vs HYBRID @6000 | +0..+2 (.69-1.0) | +5 (.180) | +2..+4 · +4 | +1..+7 / +2..+4 | +2; +1..+2 / +1..+2; +3 | +0 / +4..+8 | +7..+11; .33-.60; .31-.50 |
| HYBRID 15500 vs 6000 | −1..+0 (1.0) | +6 (.070) | +0..+1 · +4 | +0..+4 / +0..+6 | +3; +0..+1 / +5; +7 | +0..+1 / +4..+8 | +13..+16; .118-.177; .071-.106 |
| GREP_AGENT vs HYBRID_6000 | −7..−6 (.065-.109) | **+12 (.004)** | −5..−4 · +7 | – | −3; −5..−4 / +11; +14 | – (14/90 judged) | **+40..+42; .0002-.0005; .001** |
| GREP_AGENT vs HR_6000 | −7 (.092) | +4..+6 (.15-.42) | −6 · +2..+3 | – | −4; −6..−5 / +7; +6 | – | +24..+29; .015-.052; .027-.073 |
| ITER_BM25 vs HYBRID_6000 | −2..−1 (.69-1.0) | +12 (.004) | −1..0 · +6 | – | +2; +0..+1 / +11; +14 | −2 / +7..+11 | +29..+31; .004-.010; .009-.020 |
| ITER_BM25 vs HR_6000 | −2 (.727) | +4..+6 (.15-.39) | −2 · +1..+2 | – | +1; −1..0 / +7; +6 | +0 / +3..+5 | +13..+18; .11-.26; .14-.31 |
| HYBRID_AGENT vs HYBRID_6000 | +3..+4 (.34-.55) | +11 (.007) | +4..+5 · +6 | – | +7; +5..+6 / +10; +13 | – | +16..+18; .15-.21; .100-.114 |
| HYBRID_AGENT vs HR_6000 | +3 (.581) | +3..+5 (.23-.55) | +3 · +1..+2 | – | +6; +4..+5 / +6; +5 | – | 0..+4; .84-1.0; .64-1.0 |
| SETR_NOFILL vs HYBRID_3000 | +2..+3 (.51-.73) | +10..+11 (.003-.006) | +3..+4 · +8 | – | – | – | +16..+20; .068-.170; .039-.116 |
| GOLD vs HR_6000 | +7 (.039) | +5..+7 (.12-.30) | +6 · +1..+2 | – | +10; +8..+9 / +8; +7 | – | −4..0; .84-1.0; .70-1.0 |
| GOLD vs HYBRID_24000 | +6 (.109) | +5 (.180) | +6 · +1 | – | – | – | −2; 1.0; .94 |

Notes on the replication columns:
- The replicate rows of the agent, GOLD and FUSION contrasts replicate only the frozen side. The agents and GOLD are
  single runs.
- GREP_AGENT's no-contam interaction stays at p_q .003-.005 and p_c .011-.012. Against HYBRID r1 and r2 it is p_c .003
  and < .001.

### 2.2 Summary: where each contrast replicates (`lang_tables.md` "Summary" and "Pattern")

Family: 14 contrasts, so 28 within-language sign tests and 14 interaction tests. **Strict rule:** a half "shows" a
contrast if its sign test is p < .05 under every assignment of unresolved cells, with the net in the pooled direction.

| class (strict, raw p) | contrasts |
|---|---|
| replicates in both languages | **none** (Lex-RAG vs FUSION @3000 comes closest: fa +7..+8, p .039-.065; en +8, p .008) |
| English only | Lex-RAG vs FUSION @3000; GREP_AGENT vs HYBRID_6000; ITER_BM25 vs HYBRID_6000; HYBRID_AGENT vs HYBRID_6000; SETR_NOFILL vs HYBRID_3000 |
| Persian only | GOLD vs HR_6000 |
| neither | HR vs HYBRID @6000; Lex-RAG vs FUSION @6000; Lex-RAG vs HYBRID @6000; HYBRID budget; GREP_AGENT vs HR; ITER_BM25 vs HR; HYBRID_AGENT vs HR; GOLD vs HYBRID_24000 |
| after Holm over 28 | **all 14 "neither"** |

A separate rule asks whether there is a difference between the halves at all. A contrast counts as
language-dependent only if the interaction test rejects at both levels.

| pattern | contrasts |
|---|---|
| **language-dependent, survives Holm-14 at both levels** | **GREP_AGENT vs HYBRID_6000** (Holm p_q .007, p_c .019) |
| language-dependent, raw p only | ITER_BM25 vs HYBRID_6000 (raw p_c .009-.020; Holm .25; without the 26: p_c .071-.113) |
| opposite signs, interaction not shown | GREP_AGENT vs HR_6000 (fa −7, en +4..+6; p_c .027-.073) |
| same direction in both halves, no sign of interaction (T within ±4 pp) | Lex-RAG vs FUSION @6000 and @3000; HYBRID_AGENT vs HR_6000; GOLD vs HR_6000; GOLD vs HYBRID_24000 |
| English-concentrated, but the interaction is not shown (T +7..+20 pp, p_c ≥ .04) | HR vs HYBRID @6000; Lex-RAG vs HYBRID @6000; HYBRID budget; ITER_BM25 vs HR; HYBRID_AGENT vs HYBRID_6000; SETR_NOFILL vs HYBRID_3000 |

- **HR vs HYBRID.** The English concentration recurs in 4 of the 5 other readers (en +4..+7 for gpt-oss-20b, Qwen3,
  Llama; fa −4..+2). The interaction is never shown (p_c .077-.172).
- **Interaction CIs are wide.** For the selector the T interval is q [−17.8, 20.0] pp at 3,000 and [−17.8, 17.8] at
  6,000. "No interaction" means not detected, not equivalent.

### 2.3 Within-language minimum detectable effect (`lang.json` mde_generic and per-contrast mde)

Exact two-sided sign test, α = .05, power .80, n = 45:

| discordance ψ | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|---|---|---|
| MDE, points of 45 | 4.5 | 6.8 | 8.3 | 9.4 | 10.3 | 11.2 | 12.0 |
| MDE, n = 25 (English half without the 26) | 2.5 | 3.8 | 5.0 | 6.2 | 7.5 | 8.1 | 8.7 |
| MDE, n = 90 (reference) | 8.4 | 10.4 | 12.1 | 13.6 | 15.0 | 16.2 | 17.3 |

- **Observed discordance:** ψ is 0.09-0.29 in the Persian half and 0.18-0.36 in the English half. That gives
  observed-ψ MDEs of 4.1-10.1 of 45 for Persian and 7.8-11.3 for English.
- **Two cases:** HR vs HYBRID is fa ψ 0.09 → 4.1 and en ψ 0.23 → 9.0; GREP vs HYBRID is fa 0.23 → 8.9 and
  en 0.36 → 11.3.
- **The estimate of "about 8 of 45" in FRAMING is right for ψ ≈ 0.2,** that is 8.3 of 45 = 18 pp. English contrasts
  are more discordant, so they need larger effects.

### 2.4 Reader tokens per evidence character by language (`tokjudge.json` tokens)

| reader | fa tokens/char | en tokens/char | **fa/en** | 95% CI |
|---|---|---|---|---|
| z-ai/glm-5.3-flash | 0.409 | 0.228 | **1.80** | [1.73, 1.86] |
| deepseek/deepseek-v4-flash | 0.396 | 0.226 | **1.75** | [1.69, 1.81] |
| meta-llama/llama-3.3-70b-instruct | 0.348 | 0.226 | **1.53** | [1.48, 1.59] |
| openai/gpt-oss-20b | 0.341 | 0.225 | **1.51** | [1.47, 1.56] |
| google/gemma-3-27b-it | 0.354 | 0.242 | **1.46** | [1.38, 1.55] |
| qwen/qwen3-235b-a22b-2507 | 0.758 | 0.254 | **2.99** | [2.88, 3.09] |

- **GLM reproduces S7.** The GLM value is 1.795, against S7's 1.797 on 2,106 calls.
- **The spread is in Persian.** English costs 0.225-0.254 tokens per character for every tokenizer. Persian costs
  0.34-0.76, so the language gap lies almost entirely on the Persian side.
- **Whole-prompt ratios** (pooled) are a little lower: 1.44-2.86.

### 2.5 Judges by language (`tokjudge.json` judge1_vs_judge2, judges_vs_human_120, disagreement_causes)

Judge 1 vs judge 2 on the GLM cells resolved under both:

| scope | fa: n, agree %, kappa | en: n, agree %, kappa | kappa fa − en (perm p; 95% CI) |
|---|---|---|---|
| 8 cells × 96 | 380, 89.7, **0.690** | 370, 92.7, **0.817** | −0.126 (p .21; [−0.35, 0.06]) |
| same, without LAW-fa-T1-002 (the کلیمیان question) | 372, 91.4, 0.737 | 370, 92.7, 0.817 | −0.080 (p .38; [−0.29, 0.09]) |
| four 6,000 cells × 96 | 190, 91.1, 0.738 | 184, 92.4, 0.815 | −0.076 (p .45; [−0.29, 0.12]) |

The disagreement rate is fa 10.3% vs en 7.3%: +3.0 pp, p .43, CI [−4.0, 10.0]. That rate counts all 8 cells × 96.

Each judge vs the human on the 120 reviewed cells. The 30 reviewed questions are 12 Persian and 18 English.

| judge | lang | agree | kappa | stricter (judge 0, human 1) | laxer | rejected of human-accepted |
|---|---|---|---|---|---|---|
| judge 1 | fa | 33/46 | 0.419 | 13 | 0 | 13/37 (35%) |
| judge 1 | en | 46/70 | 0.327 | 24 | 0 | 24/58 (41%) |
| judge 2 | fa | 35/48 | 0.445 | 13 | 0 | 13/38 (34%) |
| judge 2 | en | 43/69 | 0.282 | 26 | 0 | 26/58 (45%) |

- **No language difference against the human:** stricter-share fa − en is −6.0 pp for judge 1 (p .61) and −10.6 pp
  for judge 2 (p .43).
- **The disagreements by cause** are 39 Persian cells (16 questions) and 27 English cells (13 questions):

| cause | fa (J1 / J2 stricter) | en (J1 / J2 stricter) |
|---|---|---|
| extra_unsupported | 18 (4 / 14) | 12 (5 / 7) |
| omission_disputed | 3 (0 / 3) | 9 (2 / 7) |
| synonym_fa (کلیمیان, one question) | 7 (7 / 0) | 0 |
| judge_misread | 5 (1 / 4) | 1 (1 / 0) |
| numeric_tolerance (one question) | 4 (1 / 3) | 0 |
| substantive_error | 2 (0 / 2) | 5 (2 / 3) |

- **Persian-specific causes:** the Persian surplus is the single synonym question (7 cells) plus one numeric-tolerance
  question (4 cells). Without the synonym question it is 32 vs 27.
- **English-heavy causes:** English has more disputed omissions (9 vs 3) and more real errors (5 vs 2).

## 3. Cost

$0.00. No API calls, no GPU, no ledger. CPU only: about 2 min for `t1_lang.py` and under 1 min for `t2_tok_judge.py`.

## 4. Failed, skipped, caveats

- **Judge 2 is a moving snapshot for the post-hoc arms.** S8's judge-2 run kept filling while this stream ran: ITER_BM25
  went from about 59 to 88 resolved cells between my first and final runs. The final tables use the 20:08 snapshot.
  - GREP_AGENT (14/90) and GOLD (1/90) were still undetermined at that point.
  - HYBRID_AGENT, HYBRID_3000, the 3,000 Lex-RAG/FUSION pair and HYBRID_24000 have no judge-2 verdicts at all.
  - Rerun `t1_lang.py` and `t3_compact.py` once judge 2 is finished.
- **Replicates are one-sided for the post-hoc arms.** The agents, GOLD and SETR_NOFILL have one run. Their "replicate"
  rows vary only the frozen control's reader draw. LEX5B_3000, FUSION_3000, HYBRID_3000 and HYBRID_24000 have no
  replicates.
- **The five other readers exist for the 8 frozen arms only.** So only HR vs HYBRID, Lex-RAG vs HYBRID and the HYBRID
  budget effect have reader rows.
- **The closed-book-correct set** (S7 contam.json) was defined from the GLM closed-book run. It is applied unchanged to
  the other readers' rows.
- **Skipped: the arm × language term in S5's question + pack GLMM** (FRAMING §2 item 2, last clause). It was not in this
  brief's analysis list, and S5's `glmm_pack.py` has no language covariate. The permutation tests above are the
  interaction tests used here.
- **What the permutation tests can and cannot show.** They condition on the observed per-question differences. Language
  is confounded with source family (components nest within language), so a significant interaction means "differs
  between the two halves". It does not mean "differs because of the language" (FRAMING §2).
- **The judge-vs-human comparison is small:** 12 Persian and 18 English questions, at 4 cells each. The language
  differences there have very low power.

## 5. Facts for the paper

1. **Only one contrast changes significantly between the halves after correction: GREP_AGENT vs HYBRID_6000.** It is
   +12/45 in the English half and −7..−6 in the Persian half (T +40..+42 pp). Component-level exact permutation gives
   p = .001 (Holm over 14 contrasts .019). The pattern holds without the 26 closed-book-correct questions (p_c ≤ .012)
   and against both HYBRID reader replicates. The BM25 agent shows the same pattern only before correction (+12 vs
   −2..−1, raw p_c .009-.020, Holm .25). Source: `S14_bilingual/lang.json` summary; `lang_tables.md`.
2. **The selector effect has the same sign in both halves, with no sign of an interaction.** Lex-RAG vs FUSION is
   +7..+8 fa / +8 en at 3,000 and +4..+6 / +5 at 6,000 (T within ±2 pp, p_c ≥ .79). It is never significant in both
   halves at once (fa p .039-.065 at 3,000). Every other single-pass contrast, and the budget effect, lies mainly in
   the English half:
   - HR vs HYBRID: +6..+8 en vs +0..+1 fa;
   - HYBRID 15,500 vs 6,000: +6 vs −1..0;
   - HYBRID_AGENT vs HYBRID: +11 vs +3..+4;
   - SETR_NOFILL vs HYBRID_3000: +10..+11 vs +2..+3.

   For none of these four is the interaction significant (p_c .039-.172). After Holm over 28 tests, no within-half
   contrast is significant. Source: `S14_bilingual/lang.json`, `compact_table.md`.
3. **A within-language paired contrast at n = 45 detects only about 8 of 45 (18 pp)** at the typical discordance
   ψ ≈ 0.2 (exact sign test, 80% power). The observed discordance gives 4.1-11.3 of 45. The English half without the
   memorisable questions (n = 25) detects about 5 of 25. Source: `S14_bilingual/lang.json` mde_generic, contrasts.*.mde.
4. **All six reader tokenizers charge more per character for Persian evidence than for English:** fa/en marginal
   tokens per evidence character 1.46 (Gemma 3) to 1.80 (GLM-5.3) for five readers, and 2.99 for Qwen3-235B (95% CIs
   within ±0.11). English costs 0.225-0.254 tokens/char for every reader and Persian 0.34-0.76. "A character budget is
   not a token budget" therefore holds for every model family tested, not only GLM. Source: `S14_bilingual/tokjudge.json`
   tokens.
5. **Judge agreement is nominally lower on Persian cells, but not significantly.** Judge-1/judge-2 kappa is 0.690 for
   Persian against 0.817 for English (difference −0.13, permutation p .21, CI [−0.35, 0.06]). Without the one کلیمیان
   question the difference falls to −0.08 (p .38). Against the human, neither judge is stricter on Persian: judge 1
   rejects 35% of human-accepted Persian cells and 41% of English ones, judge 2 34% vs 45%. Source:
   `S14_bilingual/tokjudge.json`.
