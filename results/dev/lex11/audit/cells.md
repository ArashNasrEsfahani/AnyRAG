
## V2-dev-BOOK-en-T3-003 (calculation)
expected: Franklin signed his indentures at twelve; Douglass was probably between seven and eight when he left for Baltimore. Douglass was younger, by about four to five years.
- R1 operand [exact] Franklin was twelve when he signed the indentures. (req=True)
- R2 operand [interval] Douglass was probably between seven and eight. (req=True)
- R3 derived [exact] Douglass was younger (by about 4-5 years). (req=True)
### cell 0 | AGENTIC5+doc_card,section_expand | devrun/judge/AGENTIC5+doc_card+section_expand | gemma4:e4b-it-qat
  * [R1/correct] Benjamin Franklin was twelve years old when bound as an apprentice to his brother
  * [R2/correct] Frederick Douglass was probably between seven and eight years old when sent to Baltimore
  * [R3/correct] Frederick Douglass was younger at that turning point
### cell 1 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] Benjamin Franklin was twelve years old when bound as an apprentice to his brother
  * [R2/correct] Frederick Douglass was probably between seven and eight years old when sent to Baltimore
  * [R3/correct] Benjamin Franklin was older at that turning point (so Douglass was younger)
### cell 2 | HYBRID | devrun/judge/HYBRID | gemma4:e4b-it-qat
  * [R1/correct] Benjamin Franklin was twelve years old when bound as an apprentice to his brother
  * [R2/correct] Frederick Douglass was between seven and eight years old when sent to Baltimore
  * [R3/correct] Frederick Douglass was younger at that turning point

## V2-dev-LAW-en-T3-001 (comparison)
expected: Under the Data Governance Act a data holder is a legal person (including public sector bodies and international organisations) or a natural person who is not a data subject with respect to the data in question, and the status rests on having the right to grant access to or share personal or non-personal data (Art. 2(8)). Under the Data Act a data holder is any natural or legal person, with no data-subject exclusion, and the status rests on having the right or obligation to use and make available data (Art. 2(13)).
- R1 fact [exact] DGA: a natural person is a data holder only if not a data subject with respect to the data. (req=True)
- R2 fact [exact] DGA: the status rests on a right to grant access to or share data. (req=True)
- R3 fact [exact] Data Act: any natural or legal person can be a data holder, with no data-subject exclusion. (req=True)
- R4 fact [exact] Data Act: the status rests on a right or obligation to use and make available data. (req=True)
### cell 0 | AGENTIC3 | devrun/judge/AGENTIC3 | gemma4:e4b-it-qat
  * [R1/correct] DGA: a natural person who is not a data subject can be a data holder
  * [R2/correct] DGA: the data‑holder status rests on the right to grant access to or share data
  * [R3/correct] Data Act: any natural or legal person can be a data holder
  * [R4/correct] Data Act: the status rests on a right or obligation to use and make available data
### cell 1 | AGENTIC5+doc_card,section_expand | devrun/judge/AGENTIC5+doc_card+section_expand | gemma4:e4b-it-qat
  * [R1/correct] DGA defines a data holder as a legal person … or a natural person who is not a data subject … with the right to grant access to or share data
  * [R2/correct] In the DGA the status rests on the right to grant access to or share data
  * [R3/correct] Data Act defines a data holder as any natural or legal person with the right or obligation to use and make available data
  * [R4/correct] In the Data Act the status rests on a right or obligation to use and make available data
### cell 2 | AGENTIC5 | devrun/judge/AGENTIC5 | gemma4:e4b-it-qat
  * [R1/correct] DGA defines a data holder as a legal person, including public sector bodies, or a natural person who is not a data subject
  * [R2/correct] In the DGA the status rests on having the right to grant access to or share data
  * [R3/correct] Data Act defines a data holder as any natural or legal person, with no data‑subject exclusion
  * [R4/correct] In the Data Act the status rests on having the right or obligation to use and make available data
### cell 3 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] DGA allows a natural person who is not a data subject to be a data holder.
  * [R2/correct] In DGA the status rests on the right to grant access to or share data.
  * [R3/correct] Data Act allows any natural or legal person to be a data holder.
  * [R4/correct] In Data Act the status rests on the right or obligation to use and make available data.
  * [None/duplicate] The Data Governance Act allows a natural person who is not a data subject to be a data holder, based on having the right to grant access to or share data.
### cell 4 | HYBRID | devrun/judge/HYBRID | gemma4:e4b-it-qat
  * [R1/correct] A natural person can be a data holder under the DGA only if not a data subject with respect to the specific data
  * [R2/correct] The DGA data‑holder status rests on the right to grant access to or share data
  * [R3/correct] Any natural or legal person can be a data holder under the Data Act
  * [R4/correct] The Data Act data‑holder status rests on a right or obligation to use and make available data
  * [None/duplicate] The Data Governance Act allows a natural person to be a data holder if they are not a data subject, while the Data Act defines a data holder as a natural or legal person with the right or obligation t
### cell 5 | LEX2 | devrun/judge/LEX2 | gemma4:e4b-it-qat
  * [R1/correct] DGA: a natural person who is not a data subject with the right to grant access or share data
  * [R2/correct] DGA: the status rests on a right to grant access to or share data
  * [R3/correct] Data Act: a natural or legal person with the right or obligation to use and make available data
  * [R4/correct] Data Act: the status rests on a right or obligation to use and make available data
  * [None/extra] Comparison: DGA limits natural persons to non‑data‑subjects, while the Data Act includes any natural or legal person
### cell 6 | LEX3 | devrun/judge/LEX3 | gemma4:e4b-it-qat
  * [R1/correct] Data Governance Act defines a natural person as a data holder only if not a data subject and with the right to grant access to or share data.
  * [R3/correct] Data Act defines any natural or legal person as a data holder with the right or obligation to use and make available data.

## V2-dev-LAW-en-T3-003 (calculation)
expected: Human trafficking. Bribing a foreign public official (s.6) carries up to 10 years' imprisonment, a fine or both on indictment (Bribery Act s.11(1)(b)); human trafficking (s.2) carries imprisonment for life on indictment (Modern Slavery Act s.5(1)(a)).
- R1 operand [<=] Bribery of a foreign public official: maximum 10 years on indictment. (req=True)
- R2 operand [exact] Human trafficking: imprisonment for life on indictment. (req=True)
- R3 derived [exact] Human trafficking carries the heavier maximum. (req=True)
### cell 0 | LEX3 | devrun/judge/LEX3 | gemma4:e4b-it-qat
  * [R1/correct] Bribery Act 2010 - 10 years
  * [R2/correct] Modern Slavery Act 2015 - life
  * [R3/correct] Human trafficking carries the heavier maximum penalty

## V2-dev-LAW-fa-T3-001 (calculation)
expected: Under the 1334 bill the applicant had to be at least 30 (Art. 3(b)); under the 1364 law at least 25 (Art. 9(2)): the minimum age fell by 5 years.
- R1 fact [>=] Under the 1334 Press bill the applicant had to be at least 30. (req=True)
- R2 fact [>=] Under the 1364 Press Law the applicant must be at least 25. (req=True)
- R3 derived [exact] The minimum age was lowered by 5 years. (req=True)
### cell 0 | AGENTIC3 | devrun/judge/AGENTIC3 | gemma4:e4b-it-qat
  * [R1/correct] لایحه قانونی مطبوعات 1334 - سی
  * [R2/correct] قانون مطبوعات 1364 - 25
  * [R3/correct] حداقل سن متقاضی از سی سال در لایحه قانونی مطبوعات 1334 به 25 سال در قانون مطبوعات 1364 کاهش یافته است.
### cell 1 | AGENTIC5+doc_card,section_expand | devrun/judge/AGENTIC5+doc_card+section_expand | gemma4:e4b-it-qat
  * [R1/correct] در لایحه قانونی مطبوعات 1334 سن متقاضی باید از سی سال کمتر نباشد
  * [R2/correct] در قانون مطبوعات 1364 سن متقاضی باید 25 سال باشد
  * [R3/correct] حداقل سن از 30 سال به 25 سال کاهش یافت
### cell 2 | AGENTIC5 | devrun/judge/AGENTIC5 | gemma4:e4b-it-qat
  * [R2/correct] قانون مطبوعات 1364: 25 سال
  * [R1/correct] لایحه قانونی مطبوعات 1334: سی سال
  * [R3/correct] حداقل سن متقاضی امتیاز روزنامه از سی سال در لایحه قانونی مطبوعات 1334 به 25 سال در قانون مطبوعات 1364 کاهش یافته است.
### cell 3 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] لایحه قانونی مطبوعات 1334 ماده 3: سن او از سی سال کمتر نباشد.
  * [R2/correct] قانون مطبوعات 1364 ماده 9: دارا بودن حداقل 25 سال.
  * [R3/correct] حداقل سن متقاضی از ۳۰ سال در لایحه قانونی مطبوعات ۱۳۳۴ به ۲۵ سال در قانون مطبوعات ۱۳۶۴ کاهش یافته است.
### cell 4 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R1/correct] لایحه قانونی مطبوعات 1334 ماده 3: سن او از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن متقاضی از 30 سال در لایحه 1334 به 25 سال در قانون 1364 کاهش یافت
### cell 5 | HYBRID | devrun/judge/HYBRID | gemma4:e4b-it-qat
  * [R1/correct] در لایحه قانونی مطبوعات 1334، ماده 3: سن او از سی سال کمتر نباشد.
  * [R2/correct] در قانون مطبوعات 1364، ماده 9: دارا بودن حداقل 25 سال.
  * [R3/correct] حداقل سن متقاضی از ۳۰ سال در لایحه قانونی مطبوعات ۱۳۳۴ به ۲۵ سال در قانون مطبوعات ۱۳۶۴ کاهش یافته است.
### cell 6 | HYBRID+rerank | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد.
### cell 7 | HYBRID | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] در لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد.
  * [R2/correct] در قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال.
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت (۵ سال کمتر).
### cell 8 | HYBRID+kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد.
### cell 9 | HYBRID+doc=3,kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد.
### cell 10 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن او از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن متقاضی امتیاز از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت (۵ سال کمتر)
### cell 11 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن متقاضی امتیاز از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد
### cell 12 | LEX5B | lex5b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن او از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد
### cell 13 | LEX6 | lex6/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد.
### cell 14 | LEX7 | lex7/v71/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] لایحه قانونی مطبوعات 1334 - ماده 3: سن متقاضی از سی سال کمتر نباشد
  * [R2/correct] قانون مطبوعات 1364 - ماده 9: دارا بودن حداقل 25 سال
  * [R3/correct] حداقل سن از ۳۰ سال در لایحه ۱۳۳۴ به ۲۵ سال در قانون ۱۳۶۴ کاهش یافت، یعنی ۵ سال کمتر شد

## V2-dev-LAW-fa-T3-002 (comparison)
expected: In both, the reply may not exceed twice the length of the original (Art. 9 of the 1334 bill, Art. 23 of the 1364 law). Only the 1364 law sets a deadline: the reply must be sent in writing within one month, and the paper must print it free in one of the two issues after receiving it. The 1334 bill sets no deadline for sending the reply.
- R1 fact [exact] Under the 1334 bill the reply may not exceed twice the original. (req=True)
- R2 fact [exact] Under the 1364 law the reply may not exceed twice the original. (req=True)
- R3 conclusion [<=] Only the 1364 law sets a deadline: the reply must be sent within one month. (req=True)
- R4 fact [exact] The 1334 bill sets no deadline for sending the reply. (req=True)
- R5 fact [exact] Under the 1364 law the paper must print the reply in one of the two issues after receiving it. (req=False)
### cell 0 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R2/correct] قانون مطبوعات 1364 ماده 23: جواب از دو برابر اصل تجاوز نکند
  * [R1/correct] لایحه قانونی مطبوعات 1334 ماده 9: جواب از دو برابر اصل مطلب تجاوز ننماید
  * [R3/correct] در قانون مطبوعات 1364، پاسخ باید ظرف یک ماه کتبا ارسال شود
  * [R4/correct] لایحه قانونی مطبوعات 1334 هیچ مهلت مشخصی برای ارسال پاسخ تعیین نکرده است

## V2-dev-LAW-fa-T3-003 (comparison)
expected: As a civil servant: yes, Art. 74(a) allows retirement at 60 with any length of service and the state must accept. As an insured person: no, Art. 76 of the Social Security Act also requires at least ten years of premiums before the application (men must be 60), so 8 years is not enough.
- R1 conclusion [exact] Under the Civil Service Employment Law a 60-year-old may retire with any length of service. (req=True)
- R2 operand [>=] Under the Social Security Act at least ten years of premiums are required before applying. (req=False)
- R3 conclusion [exact] Under the Social Security Act the 60-year-old with 8 years of premiums cannot yet retire. (req=True)
### cell 0 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] مستخدم رسمی مشمول قانون استخدام کشوری می‌تواند تقاضای بازنشستگی کند
  * [R1/duplicate] مستخدم رسمی مشمول قانون استخدام کشوری می‌تواند تقاضای بازنشستگی کند
  * [R3/correct] بیمه‌شده مشمول قانون تأمین اجتماعی نمی‌تواند تقاضای بازنشستگی کند
  * [None/extra] در قانون استخدام کشوری، برای بازنشستگی با سن ۶۰ سال، حداقل سابقه خدمت مشخص نشده است، اما در قانون تأمین اجتماعی، برای مرد، حداقل ۱۰ سال حق بیمه لازم است.
### cell 1 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R1/correct] مستخدم رسمی با ۶۰ سال سن می‌تواند تقاضای بازنشستگی کند
  * [R3/correct] بیمه‌شده با ۸ سال حق بیمه نمی‌تواند بازنشستگی کند چون حداقل ۱۰ سال لازم است
  * [None/extra] قانون استخدام کشوری > قانون استخدام کشوری > فصل هشتم - در بازنشستگی و وظیفه: NOT_FOUND
  * [None/extra] قانون تأمین اجتماعی > قانون تأمین اجتماعی > ماده 76: NOT_FOUND
### cell 2 | LEX3+doc_card,section_expand | devrun/judge/LEX3+doc_card+section_expand | gemma4:e4b-it-qat
  * [R1/correct] در حالت قانون استخدام کشوری، مرد ۶۰ ساله با هر سابقه خدمت می‌تواند تقاضای بازنشستگی کند.
  * [R3/correct] در حالت قانون تأمین اجتماعی، او چون حداقل ده سال حق بیمه مقرر را پرداخت نکرده است، نمی‌تواند تقاضای مستمری بازنشستگی کند.

## V2-dev-RPT-en-T2-003 (calculation)
expected: At least about 41%: over £5.4 billion of business deposits against £13.3 billion of customer balances.
- R1 operand [>=] Business deposits held at Wise were over £5.4 billion. (req=False)
- R2 operand [exact] Total customer balances were £13.3 billion. (req=False)
- R3 derived [>=] Business deposits were at least about 41% of customer balances. (req=True)
### cell 0 | AGENTIC5 | devrun/judge/AGENTIC5 | gemma4:e4b-it-qat
  * [R3/correct] The ratio of Wise Business, 2024, Total Customer Balances to Wise Business, 2024, Business Deposits is 2.463

## V2-dev-RPT-en-T3-001 (comparison)
expected: Both agreed €120 million (excluding VAT). Vodafone Turkey paid the fee in advance in one go, funded by a capital injection; Turkcell paid in instalments - an initial payment of €81.6 million (down payment plus all VAT) in 2023, about €31.1 million in April 2024, and the rest due within a year.
- R1 fact [exact] Turkcell's 2G extension fee was €120 million (plus €21.6 million VAT). (req=True)
- R2 fact [exact] Vodafone Turkey's 2G extension fee was €120 million (excluding 18% VAT). (req=True)
- R3 fact [exact] Vodafone Turkey paid in advance via a capital injection. (req=True)
- R4 fact [exact] Turkcell paid in instalments (€81.6m initial payment in 2023, €31.1m in April 2024, remainder within a year). (req=True)
- R5 conclusion [exact] The two fees were the same. (req=False)
### cell 0 | AGENTIC3 | devrun/judge/AGENTIC3 | gemma4:e4b-it-qat
  * [R1/correct] Turkcell - EUR 120 million (plus EUR 21.6 million of value-added taxes)
  * [R2/correct] Vodafone Turkey - €120 million (excluding 18% VAT)
  * [R4/correct] Turkcell's initial payment of EUR 81.6 million paid in TRY 1.7 billion during 2023
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection
### cell 1 | AGENTIC5+doc_card,section_expand | devrun/judge/AGENTIC5+doc_card+section_expand | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey - €120 million
  * [R1/correct] Turkcell - EUR 120 million (plus EUR 21.6 million of value-added taxes)
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection and signed the extension agreement effective as of 27 April 2023.
  * [R4/correct] Turkcell paid the initial payment which includes the down payment for the extension fee and total amount of value-added taxes, amounting to EUR 81.6 million in Turkish Lira at the amount of TRY 1.7 bi
### cell 2 | AGENTIC5 | devrun/judge/AGENTIC5 | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey - €120 million
  * [R1/correct] Turkcell - EUR 120 million (plus EUR 21.6 million of value-added taxes)
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection and signed the extension agreement effective as of 27 April 2023.
  * [R4/correct] Turkcell paid the initial payment which includes the down payment for the extension fee and total amount of value-added taxes, amounting to EUR 81.6 million in Turkish Lira during 2023.
### cell 3 | B2 | devrun/judge/B2 | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey fee €120 million
  * [R1/correct] Turkcell fee EUR 120 million
  * [R5/correct] Both operators agreed to a fee of €120 million for the extension
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection
  * [R4/correct] Turkcell paid the initial payment in Turkish Lira during 2023 and subsequent payments in Turkish Lira in 2024
### cell 4 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey fee €120 million
  * [R1/correct] Turkcell fee EUR 120 million plus EUR 21.6 million VAT
  * [R3/correct] Vodafone Turkey paid the fee in advance with a capital injection (agreement effective 27 April 2023)
  * [R4/correct] Turkcell paid in instalments: initial payment in 2023 and €31.1 m in April 2024
### cell 5 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey fee €120 million
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R3/correct] Vodafone Turkey paid in advance via capital injection (agreement effective 27 April 2023)
  * [R4/correct] Turkcell paid €81.6 m in 2023 and €31.1 m in April 2024 as instalments
### cell 6 | HYBRID | devrun/judge/HYBRID | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey fee €120 million
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R3/correct] Vodafone Turkey paid in advance via capital injection (agreement 27 April 2023)
  * [R4/correct] Turkcell paid instalments: €81.6 m in 2023, €31.1 m April 2024, remainder later
### cell 7 | LEX3+doc_card,section_expand | devrun/judge/LEX3+doc_card+section_expand | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey - €120 million
  * [R1/correct] Turkcell - EUR 120 million (plus EUR 21.6 million of value-added taxes)
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection and signed the extension agreement effective as of 27 April 2023.
  * [R4/correct] Turkcell made an initial payment of EUR 81.6 million (including down payment and VAT) in Turkish Lira during 2023, and approximately half of the remaining installments amounting to EUR 31.1 million we
### cell 8 | LEX3 | devrun/judge/LEX3 | gemma4:e4b-it-qat
  * [R2/correct] Vodafone Turkey - €120 million
  * [R1/correct] Turkcell - EUR 120 million
  * [R3/correct] Vodafone Turkey paid the extension fee in advance with a capital injection and signed the extension agreement effective as of 27 April 2023.
  * [R4/correct] Turkcell paid the initial payment ... EUR 81.6 million in 2023 and approx. EUR 31.1 million in April 2024.
### cell 9 | HYBRID | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R2/correct] Vodafone Turkey fee €120 million (excluding 18% VAT)
  * [R3/correct] Vodafone Turkey paid in advance via a capital injection
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R4/correct] Turkcell paid in instalments: €81.6 m in 2023, €31.1 m in Apr 2024, remainder within a year
  * [R5/correct] Both operators agreed the same €120 million fee (excluding VAT)
### cell 10 | HYBRID+kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R2/correct] Vodafone Turkey fee €120 million excluding 18% VAT
  * [R4/correct] Turkcell paid in instalments (€81.6 m in 2023, €31.1 m in Apr 2024, remainder within a year)
  * [R3/correct] Vodafone Turkey paid the fee in advance via a capital injection
### cell 11 | HYBRID+rerank | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R2/correct] Vodafone Turkey fee €120 million (excluding 18% VAT)
  * [R3/correct] Vodafone Turkey paid in advance via a capital injection
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R4/correct] Turkcell paid in instalments: €81.6 m in 2023, €31.1 m in April 2024, remainder within a year
  * [R5/correct] Both operators agreed to the same €120 million fee
### cell 12 | HYBRID+doc=3,kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Turkcell 2G extension fee €120 million plus €21.6 million VAT
  * [R2/correct] Vodafone Turkey 2G extension fee €120 million excluding 18% VAT
  * [R5/correct] Both operators agreed to the same €120 million fee (excluding VAT)
  * [R3/correct] Vodafone Turkey paid the fee in advance via a capital injection
  * [R4/correct] Turkcell paid the fee in instalments: €81.6 million in 2023, €31.1 million in April 2024, remainder within a year
### cell 13 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Turkcell 2G licence extension fee: EUR 120 million (plus EUR 21.6 million VAT)
  * [R2/correct] Vodafone Turkey 2G licence extension fee: €120 million (excluding 18% VAT)
  * [R4/correct] Turkcell paid in instalments – €81.6 m in 2023, €31.1 m in April 2024, remainder within a year
  * [R3/correct] Vodafone Turkey paid the fee in advance via a capital injection (agreement effective 27 April 2023)
### cell 14 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R4/correct] Turkcell paid in installments – €81.6 million in 2023, €31.1 million in April 2024, remainder within a year
  * [R2/correct] Vodafone Turkey fee €120 million (excluding 18% VAT)
  * [R3/correct] Vodafone Turkey paid the fee in advance via a capital injection
  * [R5/correct] Both operators agreed the same headline fee of €120 million
### cell 15 | LEX5B | lex5b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R2/correct] Vodafone Turkey fee €120 million (excluding 18% VAT)
  * [R3/correct] Vodafone Turkey paid in advance with a capital injection
  * [R1/correct] Turkcell fee EUR 120 million (plus EUR 21.6 million VAT)
  * [R4/correct] Turkcell paid in instalments – EUR 81.6 m in 2023, EUR 31.1 m in April 2024, remainder within a year
  * [R5/correct] Both operators agreed to the same €120 million fee
### cell 16 | LEX6 | lex6/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Turkcell fee €120 million plus €21.6 million VAT
  * [R4/correct] Turkcell paid in instalments – €81.6 m in 2023, €31.1 m in April 2024, remainder within a year
  * [R2/correct] Vodafone Turkey fee €120 million excluding 18 % VAT
  * [R3/correct] Vodafone Turkey paid the fee in advance via a capital injection (agreement effective 27 April 2023)
  * [R5/correct] Both operators agreed to the same headline fee of €120 million
### cell 17 | LEX7 | lex7/v71/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R2/correct] Vodafone Turkey fee €120 million (excluding 18% VAT)
  * [R3/correct] Vodafone Turkey paid in advance with a capital injection
  * [None/extra] Extension agreement effective as of 27 April 2023
  * [R1/correct] Turkcell fee EUR 120 million (plus EUR 21.6 million VAT)
  * [R4/correct] Turkcell paid in installments: EUR 81.6 m in 2023, EUR 31.1 m in April 2024, remainder within a year
  * [R5/correct] Both operators agreed to the same EUR 120 million fee

## V2-dev-RPT-en-T3-002 (comparison)
expected: It fell from €304.3 million at the end of 2022 (to €106.5 million at the end of 2023) to €59.5 million at the end of 2024 - a drop of about €244.8 million.
- R1 fact [exact] The remaining commitment was €304.3m at end-2022. (req=True)
- R2 fact [exact] The remaining commitment was €59.5m at end-2024. (req=True)
- R3 fact [exact] It was €106.5m at end-2023. (req=False)
- R4 derived [approximately] The commitment fell by about €244.8m. (req=False)
### cell 0 | LEX3 | devrun/judge/LEX3 | gemma4:e4b-it-qat
  * [R1/correct] EUR 304.3m commitment at end‑2022
  * [R2/correct] EUR 59.5m commitment at end‑2024
  * [None/extra] The commitment fell from EUR 304.3m to EUR 59.5m
### cell 1 | HYBRID | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando had €304.3 m committed to fulfilment centres at end‑2022
  * [R2/correct] Zalando had €59.5 m committed to fulfilment centres at end‑2024
  * [R4/correct] The commitment fell by €244.8 m between end‑2022 and end‑2024
### cell 2 | HYBRID+rerank | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to fulfilment centres
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to fulfilment centres
  * [R4/correct] The commitment fell by 304.3 − 59.5 = 244.8m EUR between the end of 2022 and the end of 2024.
### cell 3 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to further invest in fulfilment centers
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to further invest in fulfilment centres
  * [R4/correct] The commitment fell by 244.8m EUR (304.3m − 59.5m) between the end of 2022 and the end of 2024.
### cell 4 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to fulfilment centres
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to fulfilment centres
  * [R4/correct] The commitment fell by 244.8m EUR (304.3m − 59.5m), from EUR 304.3m at the end of 2022 to 59.5m EUR at the end of 2024.
### cell 5 | LEX5+selector=glm | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to fulfilment centres
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to fulfilment centres
  * [R4/correct] The commitment fell by 244.8m EUR (304.3m − 59.5m) between the end of 2022 and the end of 2024.
### cell 6 | LEX5 | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to further invest in fulfilment centers
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to invest a further in fulfilment centres
  * [R4/correct] The commitment fell from EUR 304.3m at the end of 2022 to 59.5m EUR at the end of 2024, a decrease of EUR 244.8m (304.3 − 59.5).
### cell 7 | HYBRID | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to fulfilment centres
  * [R2/correct] Zalando - end of 2024: EUR 59.5m committed to fulfilment centres
  * [R4/correct] The commitment fell by EUR 244.8m (304.3m − 59.5m) between the end of 2022 and the end of 2024.
### cell 8 | HYBRID+rerank | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to fulfilment centres
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to fulfilment centres
  * [R4/correct] The commitment fell by 304.3 − 59.5 = 244.8m EUR between the end of 2022 and the end of 2024.
### cell 9 | LEX5B | lex5b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando - end of 2022: EUR 304.3m committed to further invest in fulfilment centres
  * [R2/correct] Zalando - end of 2024: 59.5m EUR committed to further invest in fulfilment centres
  * [R4/correct] The commitment fell by 244.8m EUR (304.3m − 59.5m), from EUR 304.3m at the end of 2022 to 59.5m EUR at the end of 2024.
### cell 10 | LEX6 | lex6/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando end‑2022 commitment €304.3 m
  * [R2/correct] Zalando end‑2024 commitment €59.5 m
  * [R4/correct] Commitment fell by €244.8 m from 2022 to 2024
### cell 11 | LEX7 | lex7/v71/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] Zalando had €304.3 m committed to fulfilment centres at end‑2022
  * [R2/correct] Zalando had €59.5 m committed to fulfilment centres at end‑2024
  * [R4/correct] The commitment fell by €244.8 m between end‑2022 and end‑2024

## V2-dev-RPT-en-T3-003 (comparison)
expected: Yes - both were audited by KPMG AG Wirtschaftsprüfungsgesellschaft, Berlin. It was new for Zalando, which switched from EY (EY GmbH & Co. KG, Stuttgart) because of mandatory auditor rotation; Delivery Hero already had KPMG as its auditor for 2023.
- R1 fact [exact] Zalando's 2024 auditor was KPMG AG Wirtschaftsprüfungsgesellschaft, Berlin. (req=True)
- R2 fact [exact] Delivery Hero's 2024 auditor was KPMG AG Wirtschaftsprüfungsgesellschaft, Berlin. (req=True)
- R3 fact [exact] KPMG was new for Zalando in 2024, replacing EY because of rotation requirements. (req=True)
- R4 fact [exact] KPMG was not new for Delivery Hero (it was the auditor for 2023 too). (req=True)
- R5 conclusion [exact] Both used the same auditor. (req=True)
### cell 0 | AGENTIC3 | devrun/judge/AGENTIC3 | gemma4:e4b-it-qat
  * [R1/correct] Zalando's 2024 auditor is KPMG AG Wirtschaftsprüfungsgesellschaft, Berlin
  * [R2/correct] Delivery Hero's 2024 auditor is KPMG AG Wirtschaftsprüfungsgesellschaft, Berlin
  * [R5/correct] Both platforms used the same statutory auditor for 2024
  * [R3/correct] KPMG was new for Zalando in 2024
  * [R4/correct] KPMG was not new for Delivery Hero in 2024

## V2-dev-RPT-fa-T2-002 (calculation)
expected: 1.17 billion installs: 700 million app installs plus 470 million game installs.
- R1 operand [exact] App installs (excluding updates) in 1399 were 700 million. (req=False)
- R2 operand [exact] Game installs (excluding updates) in 1399 were 470 million. (req=False)
- R3 derived [approximately] Total installs of apps and games in 1399 were about 1.17 billion. (req=True)
### cell 0 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [None/extra] در سال ۱۳۹۹ کاربران بازار ۱.۶ میلیارد بار برنامه و بازی دانلود یا به‌روزرسانی کردند
  * [R1/correct] ۷۰۰ میلیون برای نصب برنامه‌ها
  * [R2/correct] ۴۷۰ میلیون برای نصب بازی‌ها
  * [R3/correct] در مجموع حدود ۱.۱۷ میلیارد بار فقط نصب
### cell 1 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R2/correct] ۴۷۰ میلیون نصب بازی در سال ۱۳۹۹
  * [R1/correct] ۷۰۰ میلیون نصب برنامه در سال ۱۳۹۹
  * [R3/correct] حدود ۱.۱۷ میلیارد نصب کل در سال ۱۳۹۹
### cell 2 | LEX5 | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] ۷۰۰ میلیون بار برنامه نصب شدند
  * [R2/correct] ۴۷۰ میلیون بار بازی نصب شدند
  * [R3/correct] در مجموع حدود ۱٫۱۷ میلیارد نصب (۱٬۱۷۰ میلیون) شد
### cell 3 | HYBRID+rerank | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [None/extra] در سال ۱۳۹۹ کاربران بازار ۱.۶ میلیارد بار نصب و بروزرسانی داشتند
  * [R1/correct] ۷۰۰ میلیون نصب برنامه
  * [R2/correct] ۴۷۰ میلیون نصب بازی
  * [R3/correct] روی هم حدود ۱.۱۷ میلیارد بار فقط نصب
### cell 4 | LEX5B | lex5b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] ۷۰۰ میلیون نصب برنامه
  * [R2/correct] ۴۷۰ میلیون نصب بازی
  * [R3/correct] حدود ۱.۱۷ میلیارد بار در سال ۱۳۹۹
### cell 5 | LEX7 | lex7/v71/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R3/correct] در سال ۱۳۹۹ کاربران بازار روی‌هم‌رفته ۱٬۱۷۰ میلیون (حدود ۱.۱۷ میلیارد) بار نصب داشتند
  * [R1/correct] ۷۰۰ میلیون نصب برنامه
  * [R2/correct] ۴۷۰ میلیون نصب بازی

## V2-dev-RPT-fa-T3-001 (comparison)
expected: From 28% in 1398 to 41% in 1400 (38% in 1399), up 13 percentage points.
- R1 fact [exact] Bots decided 28% of reviews in 1398. (req=True)
- R2 fact [exact] Bots decided 41% of reviews in 1400. (req=True)
- R3 fact [exact] The 1399 share was 38%. (req=False)
- R4 derived [exact] The share rose by about 13 percentage points. (req=False)
### cell 0 | AGENTIC3 | devrun/judge/AGENTIC3 | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات‌ها در تعیین نتیجه نهایی در سال ۱۳۹۸: 28 percent
  * [R2/correct] سهم ربات‌ها در تعیین نتیجه نهایی در سال ۱۴۰۰: 41 درصد
  * [None/duplicate] سهم ربات‌ها در سال ۱۳۹۸ برابر با 28 درصد بوده و در سال ۱۴۰۰ به 41 درصد افزایش یافته است
### cell 1 | AGENTIC5 | devrun/judge/AGENTIC5 | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات در تعیین نتیجه ... 1398: 28 درصد
  * [R2/correct] سهم ربات در تعیین نتیجه ... 1400: 41 درصد
  * [None/duplicate] سهم ربات در تعیین نتیجه در سال 1398 برابر با 28 درصد بوده و در سال 1400 به 41 درصد رسیده است.
### cell 2 | B2 | devrun/judge/B2 | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1398 - 28%
  * [R2/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1400 - 41%
  * [R4/extra] سهم ربات ها در تعیین نتیجه نهایی از 28 درصد در سال 1398 به 41 درصد در سال 1400 افزایش یافته است.
### cell 3 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1398: 28%
  * [R2/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1400: 41%
  * [None/duplicate] سهم ربات ها در تعیین نتیجهٔ نهایی در سال 1398 28 درصد بوده و در سال 1400 به 41 درصد رسیده است.
### cell 4 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری ... در سال 1398: 28 درصد
  * [R2/correct] سهم ربات ها در تصمیم گیری ... در سال 1400: 41 درصد
  * [R4/correct] سهم ربات‌ها در سال 1398 28 درصد بوده و در سال 1400 به 41 درصد افزایش یافته است.
### cell 5 | HYBRID | devrun/judge/HYBRID | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری ... سال 1398: 28%
  * [R2/correct] سهم ربات ها در تصمیم گیری ... سال 1400: 41%
  * [R4/correct] سهم ربات ها ... از 28 درصد در سال 1398 به 41 درصد در سال 1400 افزایش یافته است.
### cell 6 | LEX3+doc_card,section_expand | devrun/judge/LEX3+doc_card+section_expand | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1398: 28 درصد
  * [R2/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1400: 41 درصد
  * [None/duplicate] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1398 برابر با 28 درصد بوده و در سال 1400 به 41 درصد رسیده است.
### cell 7 | LEX3 | devrun/judge/LEX3 | gemma4:e4b-it-qat
  * [R1/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1398 - 28 درصد
  * [R2/correct] سهم ربات ها در تصمیم گیری درصد نهایی برای آگهی های بررسی شده در سال 1400 - 41 درصد
  * [R4/correct] سهم ربات ها در تصمیم گیری برای آگهی های بررسی شده از سال 1398 به 28 درصد بوده و در سال 1400 به 41 درصد افزایش یافته است.
### cell 8 | HYBRID | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] سهم ربات‌ها - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] سهم ربات‌ها - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها ... ۱۳ واحد درصد افزایش
### cell 9 | HYBRID+kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] دیوار - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] دیوار - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها از ۲۸ درصد در سال ۱۳۹۸ به ۴۱ درصد در سال ۱۴۰۰ رسید؛ یعنی ۱۳ واحد درصد افزایش.
### cell 10 | HYBRID+doc=3,kw | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] دیوار - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] دیوار - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها ... ۱۳ واحد درصد افزایش
### cell 11 | HYBRID+rerank | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] سهم ربات‌ها در ۱۳۹۸: ۲۸ درصد
  * [R2/correct] سهم ربات‌ها در ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها از ۲۸ درصد در ۱۳۹۸ به ۴۱ درصد در ۱۴۰۰ رسید (افزایش ۱۳ واحد درصدی)
### cell 12 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] 1398 - سهم ربات‌ها در تصمیم‌گیری: 28 درصد
  * [R2/correct] 1400 - سهم ربات‌ها در تصمیم‌گیری: 41 درصد
  * [R4/correct] سهم ربات‌ها از 28 درصد در سال 1398 به 41 درصد در سال 1400 رسیده است (افزایش 13 واحد درصد)
### cell 13 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] سهم ربات‌ها - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] سهم ربات‌ها - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها در تعیین نتیجهٔ نهایی بررسی آگهی‌ها از ۲۸ درصد در سال ۱۳۹۸ به ۴۱ درصد در سال ۱۴۰۰ رسید (۱۳ واحد درصد افزایش).
### cell 14 | LEX5B | lex5b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] دیوار - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] دیوار - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها از ۲۸ درصد در سال ۱۳۹۸ به ۴۱ درصد در سال ۱۴۰۰ رسید؛ یعنی ۱۳ واحد درصد افزایش.
### cell 15 | LEX6 | lex6/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] دیوار - ۱۳۹۸: ۲۸ درصد
  * [R2/correct] دیوار - ۱۴۰۰: ۴۱ درصد
  * [R4/correct] سهم ربات‌ها ... از ۲۸ درصد در سال ۱۳۹۸ به ۴۱ درصد در سال ۱۴۰۰ رسید (افزایش ۱۳ واحد درصد).
### cell 16 | LEX7 | lex7/v71/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] 1398 - سهم ربات‌ها: 28 درصد
  * [R2/correct] 1400 - سهم ربات‌ها: 41 درصد
  * [R4/correct] سهم ربات‌ها از 28 درصد در سال 1398 به 41 درصد در سال 1400 رسید (افزایش 13 واحد درصد).

## V2-dev-RPT-fa-T3-002 (calculation)
expected: From over 40 million at end-1397 (start of 1398) to over 53 million in 1400: roughly 13 million (about 33%) more.
- R1 operand [>=] Active installs were just over 40 million at the start of 1398 (end of 1397). (req=True)
- R2 operand [>=] Active installs exceeded 53 million in 1400. (req=True)
- R3 derived [approximately] The increase was about 13 million (about 33%). (req=True)
### cell 0 | B3 | devrun/judge/B3 | gemma4:e4b-it-qat
  * [R1/correct] نصب فعال بازار در پایان سال 1397: بیش از 40 میلیون کاربر فعال
  * [R2/correct] نصب فعال بازار در سال 1400: بیش از 53 میلیون کاربر فعال
  * [R3/correct] تعداد نصب فعال بازار از پایان سال 1397 تا سال 1400 تقریباً 13 میلیون کاربر افزایش یافت.
### cell 1 | HYBRID+rerank | devrun/judge/HYBRID+rerank | gemma4:e4b-it-qat
  * [R1/correct] تعداد نصب فعال کافه بازار در پایان سال 1397: 40 میلیون
  * [R2/correct] تعداد نصب فعال کافه بازار در سال 1400: 53 میلیون
  * [R3/correct] افزایش تعداد نصب فعال کافه بازار از پایان سال 1397 تا سال 1400 برابر با 13 میلیون است.
### cell 2 | HYBRID+rerank | idea_test/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] کافه‌بازار - پایان سال ۱۳۹۷: 40 میلیون
  * [R2/extra] کافه‌بازار - سال ۱۴۰۰: بیش از 53 میلیون (17.8٪ رشد نسبت به سال قبل)
  * [R3/correct] تفاوت تقریبی: 53 − 40 = حدود 13 میلیون نصب فعال بیشتر از پایان ۱۳۹۷ تا سال ۱۴۰۰.
### cell 3 | LEX10 | lex10/b/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] کافه‌بازار - پایان سال ۱۳۹۷: ۴۰ میلیون نصب فعال
  * [R2/correct] کافه‌بازار - سال ۱۴۰۰: بیش از ۵۳ میلیون نصب فعال
  * [R3/correct] تقریباً ۱۳ میلیون (۵۳ − ۴۰) بیشتر شد
### cell 4 | HYBRID15500 | lex10/b_control/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] کافه‌بازار - پایان سال ۱۳۹۷: بیش از ۴۰ میلیون نصب فعال
  * [R2/correct] کافه‌بازار - سال ۱۴۰۰: بیش از ۵۳ میلیون نصب فعال
  * [R3/correct] تعداد نصب فعال کافه‌بازار از پایان سال ۱۳۹۷ تا سال ۱۴۰۰ تقریباً ۱۳ میلیون بیشتر شد
### cell 5 | HYBRID+rerank | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] پایان ۱۳۹۷ نصب فعال 40 میلیون
  * [R2/correct] سال ۱۴۰۰ نصب فعال بیش از 53 میلیون (17.8٪ رشد نسبت به سال قبل)
  * [R3/correct] تفاوت تقریباً 13 میلیون نصب فعال بیشتر
### cell 6 | LEX5 | lex5_pilot/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] نصب فعال - پایان سال ۱۳۹۷: بیش از ۴۰ میلیون
  * [R2/correct] نصب فعال - سال ۱۴۰۰: بیش از ۵۳ میلیون
  * [R3/correct] تعداد نصب فعال ... تقریباً ۱۳ میلیون (۵۳ منهای ۴۰) بیشتر شد
### cell 7 | LEX6 | lex6/eval/judge | openrouter/z-ai/glm-5.3-flash
  * [R1/correct] بازار - پایان سال ۱۳۹۷: بیش از ۴۰ میلیون نصب فعال
  * [R2/correct] بازار - سال ۱۴۰۰: بیش از ۵۳ میلیون نصب فعال
  * [R3/correct] تعداد نصب فعال کافه‌بازار از پایان سال ۱۳۹۷ تا سال ۱۴۰۰ تقریباً ۱۳ میلیون بیشتر شد (۵۳ − ۴۰ = ۱۳ میلیون).
