"""The knowledge base: ONE persistent SQLite file that every AGENTIC3 question is answered from.

A report is ingested once, with no model call, into `work/kb/kb.sqlite`: the section tree, the text units, and a table
of TYPED facts (number with the scale applied, unit kind, currency, year, calendar, definition class, char offsets)
found by regex. Retrieval is then SQL + REGEXP over that file, with a trigram index as an indexed prefilter.

    from ragbench.kb import schema, typed
    con = schema.connect(cfg.path("work_root") / "kb" / "kb.sqlite")
    schema.ensure_schema(con)
    row = typed.normalise_fact(fact, unit_text=unit.text, doc_years=(1401, 1401))

Modules: `schema` (DDL, connection factory, versions), `typed` (the typed normaliser, section 4.4 of AGENTIC3_SPEC.md),
`test_kb` (offline tests, no model). Ingestion, query helpers, the watcher and the CLI land in M2 and M3.
"""
