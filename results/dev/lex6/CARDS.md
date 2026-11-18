# LEX6 offline cards

Built by `ragbench/approaches/lex6/cards.py` (lex6-cards-v1) from the documents alone: no question and no gold annotation was read. Model openai/gpt-oss-20b, reasoning low, pinned provider (lex5_pilot providers.json), <= 4 calls in flight. Token lengths measured with the gpt-oss-20b tokenizer (HF `openai/gpt-oss-20b` tokenizer.json, o200k_harmony). Cards guide retrieval only; they never enter reader evidence.

Method: section-aware blocks (<= 4096 input tokens per call incl. instructions; <= 3600 source tokens) cover every unit of every document (full coverage, no sampling); block call = <= 60-word block summary + <= 25-word line per marked segment (+ identity metadata from block 1, kept only when its evidence snippet occurs in the block text or its section headings). Section cards (<= 64 tokens): fitting sections from their block line; small sibling sections grouped; oversized sections by a combine call over children (bounded reduce). Document: block summaries -> bounded reduce -> final call. Full card (<= 128) and compact card (<= 64) = header [display id + verified metadata] + model text; over-length -> one regeneration at 0.7x the word target, then a token-boundary cut (status `truncated`).

## Counts, cost, time

| pool | docs | blocks | section cards | calls | failed calls | input tok | output tok | $ | wall s |
|---|---|---|---|---|---|---|---|---|---|
| RPT-fa | 97/97 | 1107 | 3274 | 1503 | 0 | 2397831 | 354320 | 0.2210 | 474.5 |
| RPT-en | 15/15 | 1309 | 3392 | 1733 | 0 | 4451368 | 233483 | 0.3466 | 360.9 |
| BOOK-fa | 9/9 | 741 | 830 | 1039 | 0 | 2457592 | 147274 | 0.1941 | 270.4 |
| BOOK-en | 11/11 | 472 | 246 | 628 | 0 | 1528171 | 61268 | 0.1162 | 130.5 |
| LAW-fa | 13/13 | 147 | 591 | 210 | 0 | 515896 | 48683 | 0.0434 | 79.2 |
| LAW-en | 10/10 | 245 | 1049 | 315 | 0 | 900155 | 57301 | 0.0716 | 81.2 |
| **all** | 155 | 4021 | 9382 | 5428 | 0 | 12251013 | 902329 | **0.9929** | 1397 |

Calls/tokens/$ = every call made (cache files, incl. superseded calls and JSON re-asks); per-document usage is in each JSONL row. Reader budget at task start 5.4421, now 6.4561. Wall = card-generation wall clock (2 docs x <= 4 calls in flight); index build times are in index/MANIFEST.json.

## Token lengths (gpt-oss tokenizer)

| lang | full card | compact card | section cards |
|---|---|---|---|
| fa | n 119, p10 77, p50 109, p90 128, max 128, mean 104.9 | n 119, p10 28, p50 36, p90 47, max 60, mean 36.7 | n 4695, p10 23, p50 38, p90 59, max 64, mean 39.2 |
| en | n 36, p10 65, p50 86, p90 109, max 122, mean 86.1 | n 36, p10 25, p50 33, p90 39, max 42, mean 32.5 | n 4685, p10 17, p50 28, p90 43, max 64, mean 29.3 |

## Status and known failures

- fa: doc card status {'ok': 108, 'truncated': 11}; section card status {'ok': 4355, 'truncated': 340}; section kinds {'subtree': 3559, 'combined': 503, 'group': 633}; metadata field status {'supported': 436, 'unknown': 151, 'unsupported': 8}; kind taken from content (not stated in the text) for 0 docs
- en: doc card status {'ok': 36}; section card status {'ok': 4628, 'truncated': 57, 'failed': 2}; section kinds {'subtree': 3603, 'combined': 564, 'group': 520}; metadata field status {'supported': 124, 'unknown': 54, 'unsupported': 2}; kind taken from content (not stated in the text) for 1 docs
- explicit failures: 2
  - RE11 {"what": "section", "key": "(r) Revenue (continued)"}
  - RE12 {"what": "section", "key": "Our Executive Committee › Key steps to date … e& strategic relationship"}
- uncovered unit intervals (failed blocks): none
- publication date/version known for 50 of 155 docs (only when stated with verbatim evidence in block 1)
- 12 block replies were missing their closing brace(s) and were completed deterministically (`json_repaired` on the block); 9 documents were re-run from the cache after this parser fix (earlier rows kept in cache/*.before-json-repair.jsonl)
- known quality issues (spot audit, not exhaustive): the model sometimes gives the period year as the publication year when the cover states only one year (e.g. pub = period = 1401), or a non-date as 'version' (RE15: the report title); a section line may restate its heading ('Table of contents ...'); compact cards sometimes repeat the title instead of adding content; failed section cards are lines the block reply omitted (<= 25% of a block's segments tolerated), their text stays searchable via the chunk index

## Examples (5 per language: compact + full + 2 section cards)

### RF17 `C.derakcloud_annual-report_1399.fa` (RPT-fa)

- compact (28 tok): [RF17] گزارش وضعیت اینترنت ۱۳۹۹ — گزارش وضعیت اینترنت ۱۳۹۹، تحلیل کاربران، حملات سرویس‌منع
- full (72 tok): [RF17] گزارش ابر دِراک از وضعیت اینترنت سال ۱۳۹۹ در ایران | گزارش سالانه | ابر دِراک | سال ۱۳۹۹ — گزارش سالانهٔ ابر دِراک، وضعیت اینترنت ایران سال ۱۳۹۹، تحلیل کاربران، عملکرد شبکه، حملات سرویس‌منع، بخش‌های معرفی و تحلیل
- section `عناوین` units [0, 5] (subtree, 26 tok): عنوان‌ها و بخش‌های کلی گزارش، شامل مقدمه، بخش اول کاربران اینترنت و بخش دوم عملکرد ابر دِراک.
- section `مقدمه` units [10, 13] (subtree, 35 tok): تحلیل حملات DDoS داخلی، تغییرات در حجم و تعداد حملات، اهمیت امنیت ابری و نیاز به مقاوم‌سازی کسب‌وکارها.

### BF02 `K.wikisource_christensen_iran-sasanian.fa` (BOOK-fa)

- compact (23 tok): [BF02] ایران در زمان ساسانیان — ترجمه، تاریخ چاپ، ساختار ساسانی
- full (88 tok): [BF02] ایران در زمان ساسانیان | کتاب ترجمه فارسی | رشید یاسمی — ترجمه، تاریخ چاپ، مقدمه، ساختار اجتماعی، سیاسی، اداری، دینی، فرهنگی، نظامی، اقتصادی، حقوقی، خانوادگی ساسانی، منابع تاریخی، ادبیات، معابد، جشن‌ها، جنگ‌ها، معاهدات، باورهای زروانی و مانوی
- section `ایران در زمان ساسانیان › سرآغاز` units [0, 5] (subtree, 33 tok): مقدمه‌ای دربارهٔ انتشار و بازبینی ترجمه فارسی کتاب، تغییرات در حواشی، حروف لاتین و تشکر از دوستان.
- section `ایران در زمان ساسانیان › مقدمه … مقدمه` units [230, 239] (group, 46 tok): مقدمه‌ای مشابه با تمرکز بر منابع یونانی، لاتین، ارمنی، سریانی و چینی، و اشاره به سفرنامه هیوئن تسیانگ و آثار دینی ساسانیان.

### LF03 `L.wikisource_commercial-code-1303.fa` (LAW-fa)

- compact (41 tok): [LF03] قانون تجارت ۱۳۰۳ (25 دلو 1303 12 فر…) — قانون تجارت ۱۳۰۳: شرکت‌ها، دلالی، ورشکستگی
- full (114 tok): [LF03] قانون تجارت 1303 | قانون | 25 دلو 1303 12 فروردین و 12 خرداد 1304 شمسی — قانون تجارت ۱۳۰۳: تجار، شرکت‌های سهامی، ضمانتی، مختلط، تعاونی؛ دلالی، برات، فته‌طلب، چک؛ تصفیه، ورشکستگی، مجمع عمومی طلبکارها؛ 25 دلو 1303 12 فروردین و 12 خرداد 1304 شمسی
- section `قانون تجارت 1303 › باب اول - تجار و معاملات تجارتی` units [1, 22] (subtree, 17 tok): باب اول: تعریف تاجر، انواع معاملات تجارتی، و شرایط اعتبار تجارتی
- section `قانون تجارت 1303 › فصل سوم - در تعیین عضو ناظر` units [372, 375] (subtree, 40 tok): ماده‌های ۲۵۳ تا ۲۵۶ درباره تعیین عضو ناظر، وظایف، گزارش‌گیری، شکایت و تغییر عضو ناظر در ورشکستگی.

### RF49 `C.karnameh_annual-report_1401.fa` (RPT-fa)

- compact (30 tok): [RF49] گزارش سالانه کارنامه ۱۴۰۱ — بازار خودرو 1401، عرضه، تقاضا، قیمت، سهم معاملات
- full (87 tok): [RF49] گزارش سالانه کارنامه ۱۴۰۱ | گزارش سال کارنامه | شرکت کارنامه | سال 1401 — بازار خودرو سال 1401، عرضه، تقاضا، قیمت، سهم معاملات، شاخص‌های قیمت، تورم، بورس، سایپا، ایران‌خودرو، خودروهای داخلی و چینی، پراکندگی قیمت، سلامت خودرو، صفر ایراد، نمونه‌های خودرو
- section `فهرست` units [0, 3] (subtree, 13 tok): فهرست مطالب و بخش‌های گزارش را نشان می‌دهد.
- section `10 خودرو با بیشترین افزایش قیمت در سال 1401` units [177, 187] (subtree, 33 tok): 10 خودرو با بیشترین افزایش قیمت در 1401، قیمت‌های فروردین و اسفند و درصد رشد آن‌ها در جدول آورده شده است.

### BF05 `K.wikisource_kasravi_khuzestan-500-years.fa` (BOOK-fa)

- compact (43 tok): [BF05] تاریخ پانصد ساله خوزستان (تهران 1312) — سید محمد مشعشع، شیخ خزعل خان، حاکمیت‌های 861–1275
- full (103 tok): [BF05] تاریخ پانصد ساله خوزستان | احمد کسروی | تهران 1312 | پانصد ساله خوزستان — سید محمد مشعشع، شیخ خزعل خان، حاکمیت‌های 861–1275، جنگ‌های ایران–انگلیس، ساخت سدها، پل‌ها، حوادث 1737–1744، تاریخچهٔ کعبیان، حوادث 1909، حاکمیت‌های صفوی و قاجار
- section `تاریخ پانصد ساله خوزستان › مقدمه` units [0, 14] (subtree, 35 tok): مقدمهٔ سفر، شیخ خزعل خان، ادعاهای سید محمد مشعشع، تاریخ کعبیان، هدف نوشتن تاریخ خوزستان
- section `تاریخ پانصد ساله خوزستان › بخش نخست › شوریدن مولا مطلّب` units [400, 407] (subtree, 45 tok): شوریدن مولا مطلّب، نقش او در شورش‌های خوزستان، تعامل با عرب، و تأثیرات جنگی در سال‌های ۱۱۶۰–۱۱۷۶.

### RE03 `D.deliveryhero_annual-report_2024.en` (RPT-en)

- compact (32 tok): [RE03] Delivery Hero Annual Report — Delivery Hero 2024 report: governance, ESG, risk, financials, audit, non‑financial disclosures.
- full (68 tok): [RE03] Delivery Hero Annual Report 2024 | annual report | Delivery Hero — Delivery Hero SE 2024 report covers governance, supervisory board composition, capital increases, ESG targets, sustainability strategy, risk management, market outlook, financial performance, liquidity, capital structure, consolidated financial statements, auditor reports, and non‑financial disclosures.
- section `an amazing experience` units [0, 0] (subtree, 17 tok): Title page with heading 'an amazing experience' and 'Annual Report 2024'.
- section `Combined Management Report › Content` units [3657, 3664] (subtree, 29 tok): Outlines risk strategy updates, risk profile overviews, governance status, monthly subsidiary financial reporting, consolidation process, and central accounting for complex topics.

### BE02 `K.gutenberg-1228_darwin_origin-of-species-1st-ed.en` (BOOK-en)

- compact (36 tok): [BE02] On the Origin of Species (1859, First Edition) — Darwin’s natural selection theory, variation, fossils, geography, classification, hybridism, sexual selection
- full (89 tok): [BE02] On the Origin of Species | book | Charles Darwin | 1859, First Edition — Variation, domestication, natural selection, species concepts, ecological limits, sexual selection, hybridism, fossil record gaps, geographic distribution, classification, embryology, rudimentary organs, island biogeography, evolutionary mechanisms, Darwin, Wallace, Agassiz, fossil examples, plant and animal breeds, competition, extinction, geological time.
- section `On the Origin of Species` units [0, 2] (subtree, 15 tok): Title page listing editions and file numbers for the 1859 first edition.
- section `On the Origin of Species › By Charles Darwin, M.A., … INTRODUCTION.` units [3, 10] (group, 17 tok): Introduction by Charles Darwin with biographical notes, quotes, and a brief contents list.

### LE02 `L.eurlex-32022R0868_data-governance-act.en` (LAW-en)

- compact (34 tok): [LE02] Regulation (EU) 2022/868 — Public sector data re‑use, intermediation, altruism, fees, EU Data Innovation Board
- full (115 tok): [LE02] REGULATION (EU) 2022/868 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL | Regulation | European Parliament and the Council of the European Union — Regulation (EU) 2022/868 covers public sector data re‑use, exclusions for cultural/educational bodies, competition law compliance, data intermediation services, data altruism organisations, fees, competent bodies, information points, security, data protection, trade secrets, personal data safeguards, third‑country transfers, European Data Innovation Board, and enforcement mechanisms.
- section `REGULATION (EU) 2022/868 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL` units [0, 0] (subtree, 16 tok): Title and official journal citation of Regulation (EU) 2022/868.
- section `(Text with EEA relevance) › CHAPTER III — Requirements applicable to data intermediation services › Article 11 — Notific` units [188, 212] (subtree, 23 tok): Article 11 details notification procedures, jurisdiction, legal representative, information required, fees, and register updates for providers.

### RE08 `D.mowi_integrated-annual-report_2024.en` (RPT-en)

- compact (26 tok): [RE08] Mowi Integrated Annual Report 2024 — Mowi 2024 integrated annual performance with ESG and financial data
- full (122 tok): [RE08] Mowi Integrated Annual Report 2024 | Integrated Annual Report | Mowi — Mowi 2024 integrated annual performance: company overview, business areas, financial highlights, operational EBIT, NIBD/ROCE, EPS, ROE, ESG Index, energy, GHG, waste, escape incidents, fish welfare, antimicrobial use, freshwater stewardship, benthic impact, wildlife interactions, biodiversity projects, sustainable feed, food safety audits, employee stats, turnover, new hires, promotions, unionisation, training, health & safety, whistleblowing, community engagement, South Pole expedition.
- section `Leading the Blue Revolution` units [0, 12] (subtree, 10 tok): Table of contents with section headings and page numbers.
- section `E4 Biodiversity and ecosystem › Risks › Actions related to wildlife interaction` units [3018, 3019] (subtree, 29 tok): Actions related to wildlife interaction: monitoring, preventive measures, bird nets, predator exclusion nets, mortality registration, wild salmonid projects, restoration efforts.

### BE06 `K.gutenberg-20203_franklin_autobiography.en` (BOOK-en)

- compact (39 tok): [BE06] Autobiography of Benjamin Franklin (1916) — Franklin’s life, printing, moral virtues, public service, inventions, military, letters, 1724‑1729
- full (91 tok): [BE06] Autobiography of Benjamin Franklin | autobiography | Benjamin Franklin | 1916 — Franklin’s autobiography details his Boston upbringing, apprenticeship, printing work, London trip (1724‑25), Philadelphia ventures, moral virtues, Poor Richard’s Almanac, public service, inventions (stove), militia roles, political conflicts, and letters to Madame Brillon and Samuel Mather. It contains essays, illustrations, editorial notes, and a bibliography.
- section `B E N J A M I N` units [0, 0] (subtree, 13 tok): Title heading and a quoted anecdote about Franklin’s court visits.
- section `F R A N K L I N › ALBANY PLAN OF UNION` units [558, 565] (subtree, 21 tok): Franklin presents Albany Union Plan, debates with colonies, and its rejection by assemblies and Board of Trade.

## Files

- `<pool>.jsonl`: one row per document (metadata with evidence/status, blocks with unit intervals, section cards keyed by section path with unit intervals, full/compact cards, failures, uncovered, usage, provenance)
- `cache/<pool>.calls.jsonl`: every model call (resume cache)
- `index/<pool>.npz` (full, compact, sections, centroid: L2-normalised bge-m3 vectors; centroid = mean of the doc's chunk vectors from the prebuilt HYBRID index, for S5) + `index/<pool>.keys.json` + `index/<pool>.bm25_full`, `.bm25_sections` (bm25s, B2 tokenizer over normalize()d text), `MANIFEST.json`
- `state.json`, `PROGRESS.md`
