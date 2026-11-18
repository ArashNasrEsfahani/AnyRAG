# SUPERSEDED: first, asymmetric human-review merge (historical artifact)

These files are the first human/machine merge of the true test (2026-10-05). The first review covered only the
pipelines of each machine-discordant pair: LEX5B got 30 reviewed cells, while the controls got 15-17. The judge was
never laxer than the human, so this asymmetry favoured LEX5B. It produced an apparently significant LEX5B-vs-HYBRID
result.

**That claim is withdrawn.** The 42 missing cells were then reviewed blind. The merge reported in the paper is
`../TEST_RESULTS_HUMAN_SYM.md` / `../test_results_human_sym.json`. There, LEX5B vs. HYBRID is +5 (7/2, Holm p = .36).

These files are kept for transparency, and because `code/test_run/human_merge_sym.py` reproduces them cell for cell
before it adds the completion verdicts. Do not cite their numbers as results. See the paper, Appendix D.
