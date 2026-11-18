# Final consolidation log (2026-09-25)

Backups of the six files before this pass: `<name>.pre_final.jsonl`.
Scripts: session scratchpad `scratchpad/final/`.

## Step 0 - backups
- Saved rpt_dev, rpt_test, book_dev, book_test, law_dev, law_test as *.pre_final.jsonl.

## Step 1 - Persian review of LAW fa (persian_review_law.md) + extra rules
Applied all high/medium findings, and the low ones judged right (all low ones were accepted). Extra rules 1-8 swept over all LAW fa records; rules 1-3, 5, 6, 8 swept over RPT/BOOK fa: rules 1-3, 6 found nothing to change there (every 'چه کسی' in RPT/BOOK asks for a person, not an institution); rule 5 fixed number words sitting next to digits in the answer's own voice (idiomatic 'هر دو', 'یکی از', 'یک بار', compounds like 'یک‌ماهه' and quoted source wording left). Rule 8 checked against the statute wording for every LAW modal verb touched (Arts. 42 SSA, 6 Coop., 16 Reg., 6 Comm.).
Evidence quotes untouched (except the new E13).

| # | file | id | field | old | new | source |
|---|---|---|---|---|---|---|
| 1 | law_dev | V2-dev-LAW-fa-T1-001 | question | جدا از خود حق بیمه | علاوه بر خود حق بیمه | persian_review_law medium |
| 2 | law_dev | V2-dev-LAW-fa-T1-001 | expected_answer | مبلغی برابر نیم در هزار مبلغ عقب‌افتاده برای هر روز تأخیر، | برای هر روز تأخیر، نیم در هزارِ مبلغ عقب‌افتاده | persian_review_law low (accepted) |
| 3 | law_dev | V2-dev-LAW-fa-T1-001 | expected_answer | این خسارت هم به ترتیب ماده ۵۰ وصول می‌شود | این خسارت نیز به روش مقرر در ماده ۵۰ وصول می‌شود | persian_review_law medium; rule 4 |
| 4 | law_dev | V2-dev-LAW-fa-T1-002 | question | دست‌کم و حداکثر چقدر است | حداقل و حداکثر چقدر است | persian_review_law medium; rule 6 |
| 5 | law_dev | V2-dev-LAW-fa-T1-002 | question | چه کسی مدت آن را تعیین می‌کند | چه مرجعی مدت آن را تعیین می‌کند | persian_review_law medium; rule 2 |
| 6 | law_dev | V2-dev-LAW-fa-T1-002 | question_gloss_en | and who sets it? | and which authority sets it? | rule 2 (gloss) |
| 7 | law_dev | V2-dev-LAW-fa-T1-002 | expected_answer | ولی در هیچ حالتی کمتر از شش ماه و بیشتر از دو سال نیست | ولی این مدت در هیچ حالتی نباید کمتر از ۶ ماه یا بیشتر از ۲ سال باشد | persian_review_law medium; rule 5 digits |
| 8 | law_dev | V2-dev-LAW-fa-T1-003 | question | جلوی خروج یکی از اعضایش را بگیرد | مانع خروج یکی از اعضایش شود | persian_review_law medium |
| 9 | law_dev | V2-dev-LAW-fa-T1-003 | expected_answer | نه؛ خروج عضو اختیاری است | خیر؛ خروج عضو اختیاری است | persian_review_law low; rule 6 |
| 10 | law_dev | V2-dev-LAW-fa-T1-003 | expected_answer | بهای سهم یا سهام او حداکثر به ارزش اسمی و ظرف یک سال از تاریخ خروج، نقداً پرداخت می‌شود | بهای سهم یا سهام او باید حداکثر به ارزش اسمی و ظرف ۱ سال از تاریخ خروج نقداً پرداخت شود | persian_review_law medium; rule 8 (statute: باید ... پرداخت گردد); rule 5 |
| 11 | law_dev | V2-dev-LAW-fa-T1-003 | expected_answer_gloss_en | is paid in cash within one year of leaving | must be paid in cash within one year of leaving | rule 8 (gloss) |
| 12 | law_dev | V2-dev-LAW-fa-T2-001 | question | چقدر وقت دارد اعتراض کند؟ | در چه مهلتی می‌تواند اعتراض کند؟ | persian_review_law medium |
| 13 | law_dev | V2-dev-LAW-fa-T2-001 | question | می‌تواند تجدیدنظر بخواهد؟ در چه مهلتی؟ | آیا می‌تواند تقاضای تجدیدنظر کند و در چه مهلتی؟ | persian_review_law low (accepted) |
| 14 | law_dev | V2-dev-LAW-fa-T2-001 | expected_answer | باید ظرف ۳۰ روز از ابلاغ، اعتراضش را کتباً به سازمان بدهد | می‌تواند ظرف ۳۰ روز از تاریخ ابلاغ، اعتراض خود را کتباً به سازمان تسلیم کند | persian_review_law medium; rules 1, 8 (Art. 42: می‌تواند ... تسلیم نماید) |
| 15 | law_dev | V2-dev-LAW-fa-T2-001 | expected_answer_gloss_en | He must lodge a written objection | He may lodge a written objection | rule 8 (gloss) |
| 16 | law_dev | V2-dev-LAW-fa-T2-002 | question | مستخدم رسمیِ ۶۱ ساله با ۱۲ سال سابقه، | مستخدم رسمیِ ۶۱ ساله با ۱۲ سال سابقهٔ خدمت | persian_review_law low; rule 7 |
| 17 | law_dev | V2-dev-LAW-fa-T2-002 | question | دو ماه بدون اطلاع و بدون عذر موجه سر کار نیامده است | دو ماه بدون اطلاع و بدون عذر موجه در محل خدمت حاضر نشده است | persian_review_law medium |
| 18 | law_dev | V2-dev-LAW-fa-T2-002 | expected_answer | بازنشسته می‌شود، نه مستعفی. | بازنشسته می‌شود و مستعفی شناخته نمی‌شود. | persian_review_law low (accepted) |
| 19 | law_dev | V2-dev-LAW-fa-T2-002 | expected_answer | غیبت دو ماهه بدون اطلاع و عذر موجه، مستخدم را از تاریخ ترک خدمت مستعفی می‌شناسد | مستخدمی که دو ماه بدون اطلاع و عذر موجه غیبت کند از تاریخ ترک خدمت مستعفی شناخته می‌شود | persian_review_law high |
| 20 | law_dev | V2-dev-LAW-fa-T2-002 | expected_answer | اما ماده ۶۵ کسی را که مشمول ماده ۷۴ یا ۷۵ باشد استثنا کرده و بازنشسته می‌کند | اما ماده ۶۵ مستخدمِ مشمول مواد ۷۴ یا ۷۵ را از این قاعده مستثنا کرده است؛ چنین مستخدمی بازنشسته می‌شود | persian_review_law medium |
| 21 | law_dev | V2-dev-LAW-fa-T2-002 | expected_answer | این مستخدم بیش از شصت سال دارد | این مستخدم بیش از ۶۰ سال دارد | rule 5 |
| 22 | law_dev | V2-dev-LAW-fa-T2-002 | expected_answer | با هر قدر سابقه خدمت می‌تواند | با هر میزان سابقهٔ خدمت می‌تواند | persian_review_law low; rule 4 |
| 23 | law_dev | V2-dev-LAW-fa-T2-003 | question | یک شرکت تعاونی مسکن کارمندی | یک شرکت تعاونی مسکن کارمندان | persian_review_law medium |
| 24 | law_dev | V2-dev-LAW-fa-T2-003 | question | آن بخشِ سود سهام با چه نرخی و پس از کسر چه مبلغی مالیات می‌خورد؟ | بخشی که به‌عنوان سود سهام تقسیم می‌شود، با چه نرخی و پس از کسر چه مبلغی مشمول مالیات است؟ | persian_review_law high |
| 25 | law_dev | V2-dev-LAW-fa-T2-003 | expected_answer | «مالیات شرکت» ۱۰ درصد است | «مالیات شرکت» ۱۰ درصدِ درآمد مشمول مالیات است | persian_review_law low (accepted; Art. 107 opening: از درآمد مشمول مالیات ... 10%) |
| 26 | law_dev | V2-dev-LAW-fa-T2-003 | expected_answer | مبلغی برابر شش درصد سرمایهٔ منظورشده | مبلغی برابر ۶ درصد سرمایهٔ منظورشده | persian_review_law low; rule 5 |
| 27 | law_dev | V2-dev-LAW-fa-T2-003 | expected_answer_gloss_en | The company tax is 10% (Art. 107, opening) | The company tax is 10% of taxable income (Art. 107, opening) | gloss follows answer |
| 28 | law_dev | V2-dev-LAW-fa-T3-001 | expected_answer | در لایحهٔ ۱۳۳۴ سن درخواست‌کننده نباید از ۳۰ سال کمتر می‌بود | در لایحهٔ ۱۳۳۴ حداقل سن متقاضی ۳۰ سال بود | persian_review_law medium |
| 29 | law_dev | V2-dev-LAW-fa-T3-001 | expected_answer | در قانون ۱۳۶۴ متقاضی دست‌کم باید ۲۵ سال داشته باشد | در قانون ۱۳۶۴ متقاضی باید حداقل ۲۵ سال سن داشته باشد | rule 6 (statute: حداقل 25 سال); rule 8 |
| 30 | law_dev | V2-dev-LAW-fa-T3-002 | question | کدام‌یک برای فرستادن پاسخ مهلت گذاشته است | کدام‌یک برای فرستادن پاسخ مهلت تعیین کرده است | persian_review_law medium |
| 31 | law_dev | V2-dev-LAW-fa-T3-002 | expected_answer | از دو برابر اصل مطلب بیشتر باشد | از دو برابر مطلب اصلی بیشتر باشد | persian_review_law medium; rule 4 |
| 32 | law_dev | V2-dev-LAW-fa-T3-002 | expected_answer | فقط قانون ۱۳۶۴ مهلت گذاشته است | فقط قانون ۱۳۶۴ مهلت تعیین کرده است | persian_review_law medium |
| 33 | law_dev | V2-dev-LAW-fa-T3-002 | expected_answer | پاسخ باید ظرف یک ماه کتباً | پاسخ باید ظرف ۱ ماه کتباً | rule 5 |
| 34 | law_dev | V2-dev-LAW-fa-T3-003 | question | مردی ۶۰ ساله فقط ۸ سال سابقه دارد | مردی ۶۰ ساله فقط ۸ سال سابقهٔ کار دارد | persian_review_law medium |
| 35 | law_dev | V2-dev-LAW-fa-T3-003 | question | ، و اگر بیمه‌شدهٔ قانون تأمین اجتماعی ۱۳۵۴ باشد که ۸ سال حق بیمه پرداخته | ، یا اگر به‌جای آن بیمه‌شدهٔ مشمول قانون تأمین اجتماعی ۱۳۵۴ باشد و ۸ سال حق بیمه پرداخته باشد | persian_review_law high |
| 36 | law_dev | V2-dev-LAW-fa-T3-003 | question | در هر حالت آیا می‌تواند | در هر یک از این دو حالت، آیا می‌تواند | persian_review_law low (accepted) |
| 37 | law_dev | V2-dev-LAW-fa-T3-003 | expected_answer | با دست‌کم شصت سال سن و «با هر قدر سابقه خدمت» می‌تواند | با حداقل ۶۰ سال سن و با هر میزان سابقهٔ خدمت («با هر قدر سابقه خدمت») می‌تواند | rules 4, 5, 6 |
| 38 | law_dev | V2-dev-LAW-fa-T3-003 | expected_answer | علاوه بر سن شصت سال تمام برای مرد | علاوه بر سن ۶۰ سال تمام برای مرد | rule 5 |
| 39 | law_dev | V2-dev-LAW-fa-T3-003 | expected_answer | دست‌کم ده سال حق بیمه پیش از تقاضا | دست‌کم ۱۰ سال حق بیمه پیش از تقاضا | persian_review_law low; rule 5 |
| 40 | law_dev | V2-dev-LAW-fa-AB-001 | expected_answer | این قانون درج این نام‌ها را الزامی می‌کند | این قانون درج نام صاحب امتیاز و مدیر مسئول را الزامی می‌کند | persian_review_law low (accepted) |
| 41 | law_dev | V2-dev-LAW-fa-AB-001 | expected_answer | اما مبلغی برای جریمه نمی‌گوید؛ بنابراین نمی‌توان عددی داد | اما مبلغی برای جزای نقدی تعیین نمی‌کند؛ بنابراین نمی‌توان رقمی ذکر کرد | persian_review_law medium; rule 3 |
| 42 | law_test | V2-test-LAW-fa-T1-001 | question | بر سر حضانت اختلاف پیش بیاید، چه کسی تصمیم می‌گیرد | بر سر حضانت اختلاف پیش بیاید، کدام مرجع تصمیم می‌گیرد | persian_review_law medium; rule 2 |
| 43 | law_test | V2-test-LAW-fa-T1-001 | question_gloss_en | and who decides if a dispute arises after age seven? | and which authority decides if a dispute arises after age seven? | rule 2 (gloss) |
| 44 | law_test | V2-test-LAW-fa-T1-001 | expected_answer | حضانت با رعایت مصلحت کودک به تشخیص دادگاه است | دادگاه با رعایت مصلحت کودک دربارهٔ حضانت تصمیم می‌گیرد | persian_review_law low (accepted) |
| 45 | law_test | V2-test-LAW-fa-T1-002 | question | اقلیت‌های دینی روی هم چند نماینده | اقلیت‌های دینی در مجموع چند نماینده | persian_review_law medium |
| 46 | law_test | V2-test-LAW-fa-T1-002 | expected_answer | پنج نماینده: زرتشتیان یک نماینده، کلیمیان یک نماینده، مسیحیان آشوری و کلدانی روی هم یک نماینده، و مسیحیان ارمنیِ جنوب و شمال هر کدام یک نماینده | ۵ نماینده: زرتشتیان ۱ نماینده، کلیمیان ۱ نماینده، مسیحیان آشوری و کلدانی مجموعاً ۱ نماینده و مسیحیان ارمنیِ جنوب و شمال هر کدام ۱ نماینده | persian_review_law medium; rule 5 |
| 47 | law_test | V2-test-LAW-fa-T1-003 | question | دادخواستش را به کجا باید بدهد؟ | دادخواستش را باید به کدام مرجع تسلیم کند؟ | persian_review_law medium; rules 1, 2 |
| 48 | law_test | V2-test-LAW-fa-T1-003 | expected_answer | دادخواست مستقیماً به اداره یا دایره یا شعبهٔ ثبتی داده می‌شود | دادخواست مستقیماً به اداره یا دایره یا شعبهٔ ثبتی تسلیم می‌شود | persian_review_law medium; rule 1 |
| 49 | law_test | V2-test-LAW-fa-T1-003 | expected_answer |  و در برابر آن رسید داده می‌شود | ؛ در برابر دادخواست باید رسید داده شود | persian_review_law medium; rule 8 (Art. 16: باید رسید داده شود) |
| 50 | law_test | V2-test-LAW-fa-T1-003 | expected_answer_gloss_en | which issues a receipt | which must give a receipt for it | rule 8 (gloss) |
| 51 | law_test | V2-test-LAW-fa-T1-004 | question | هر کدام چند سال اعتبار دارند | هر کدام چند سال اعتبار دارد | persian_review_law medium |
| 52 | law_test | V2-test-LAW-fa-T1-004 | question | آیا پروانهٔ موقت را می‌شود دوباره صادر کرد؟ | آیا می‌توان پروانهٔ موقت را دوباره صادر کرد؟ | persian_review_law medium |
| 53 | law_test | V2-test-LAW-fa-T1-004 | expected_answer | پروانهٔ کسب موقت یک سال و پروانهٔ کسب دائم پنج سال اعتبار دارد | پروانهٔ کسب موقت ۱ سال و پروانهٔ کسب دائم ۵ سال اعتبار دارد | rule 5 |
| 54 | law_test | V2-test-LAW-fa-T1-005 | question | از پرداخت دیونش باز مانده بوده | از پرداخت دیونش بازمانده باشد | persian_review_law medium |
| 55 | law_test | V2-test-LAW-fa-T1-005 | expected_answer | تا یک سال بعد از مرگ او | تا ۱ سال بعد از مرگ او | rule 5 |
| 56 | law_test | V2-test-LAW-fa-T2-001 | question | اگر ورثه حاضر به پرداخت سهم او نشوند، چه می‌شود؟ | اگر ورثه حاضر به پرداخت قیمت سهم او نشوند، او چه حقی دارد؟ | persian_review_law medium |
| 57 | law_test | V2-test-LAW-fa-T2-001 | question_gloss_en | and what if the heirs refuse to pay her? | and what right does she have if the heirs refuse to pay the value of her share? | gloss follows question |
| 58 | law_test | V2-test-LAW-fa-T2-001 | expected_answer | سهم زن یک‌هشتم است، چون شوهر فرزند دارد | سهم زن یک‌هشتم است، چون شوهر فرزند داشته است | persian_review_law medium |
| 59 | law_test | V2-test-LAW-fa-T2-001 | expected_answer | یک‌هشتمِ خودِ مال را می‌برد | یک‌هشتمِ خودِ مال را به ارث می‌برد | persian_review_law medium |
| 60 | law_test | V2-test-LAW-fa-T2-002 | question | تا این مرحله تأیید شود | تا مصوبه از این جهت تأیید شود | persian_review_law high |
| 61 | law_test | V2-test-LAW-fa-T2-002 | expected_answer | شورا شش فقیه و شش حقوق‌دان دارد (اصل ۹۱)، یعنی دوازده عضو. | شورا ۱۲ عضو دارد: ۶ فقیه و ۶ حقوق‌دان (اصل ۹۱). | persian_review_law medium; rule 5 |
| 62 | law_test | V2-test-LAW-fa-T2-002 | expected_answer | پس برای احکام اسلام دست‌کم ۴ نفر از ۶ فقیه لازم است | پس برای تشخیص عدم مغایرت با احکام اسلام، رأی دست‌کم ۴ نفر از ۶ فقیه لازم است | persian_review_law medium |
| 63 | law_test | V2-test-LAW-fa-T2-002 | expected_answer | برای قانون اساسی دست‌کم ۷ رأی از ۱۲ عضو لازم است | برای تشخیص عدم تعارض با قانون اساسی دست‌کم ۷ رأی از ۱۲ عضو لازم است | persian_review_law medium |
| 64 | law_test | V2-test-LAW-fa-T2-003 | question | با یک قولنامهٔ عادی فروخته شده | با قولنامهٔ عادی فروخته شده | persian_review_law low (accepted) |
| 65 | law_test | V2-test-LAW-fa-T2-004 | question | ولی صاحبش آن را تمدید نکرده است | ولی صاحب آرایشگاه پروانه را تمدید نکرده است | persian_review_law medium |
| 66 | law_test | V2-test-LAW-fa-T2-004 | expected_answer | جریمهٔ نقدی آن از یک میلیون تا ۲۵۰ میلیون ریال | جریمهٔ نقدی آن از ۱ میلیون تا ۲۵۰ میلیون ریال | rule 5 |
| 67 | law_test | V2-test-LAW-fa-T2-005 | question | با کمترین سرمایهٔ مجاز تأسیس کنند | با حداقل سرمایهٔ مجاز تأسیس کنند | persian_review_law medium; rule 6 |
| 68 | law_test | V2-test-LAW-fa-T2-005 | question | این کمترین سرمایه چقدر است | این حداقل سرمایه چقدر است | persian_review_law medium; rule 6 |
| 69 | law_test | V2-test-LAW-fa-T2-005 | expected_answer | کمترین سرمایهٔ شرکت سهامی عام هنگام تأسیس پنج میلیون ریال | حداقل سرمایهٔ شرکت سهامی عام هنگام تأسیس ۵ میلیون ریال | persian_review_law medium; rules 5, 6 |
| 70 | law_test | V2-test-LAW-fa-T2-005 | expected_answer | دست‌کم ۲۰ درصد سرمایه را خودشان تعهد کنند، یعنی یک میلیون ریال، | دست‌کم ۲۰ درصد سرمایه (یعنی ۱ میلیون ریال) را خودشان تعهد کنند | persian_review_law medium; rule 5 |
| 71 | law_test | V2-test-LAW-fa-T2-005 | expected_answer | دست‌کم ۳۵ درصدِ مبلغ تعهدشده، یعنی ۳۵۰ هزار ریال، را | دست‌کم ۳۵ درصدِ مبلغ تعهدشده (یعنی ۳۵۰ هزار ریال) را | persian_review_law medium |
| 72 | law_test | V2-test-LAW-fa-T3-001 | question | چه کسی باید امضا کند | چه مقامی باید امضا کند | rule 2 (signatory is an official) |
| 73 | law_test | V2-test-LAW-fa-T3-001 | question | پاسخ را در قانون تجارت ۱۳۰۳ و قانون تجارت فعلی (۱۳۱۱) مقایسه کنید | قانون تجارت ۱۳۰۳ و قانون تجارت فعلی (۱۳۱۱) را در این باره مقایسه کنید | persian_review_law medium |
| 74 | law_test | V2-test-LAW-fa-T3-001 | question_gloss_en | Who must initial | Which official must initial | rule 2 (gloss) |
| 75 | law_test | V2-test-LAW-fa-T3-001 | expected_answer | دفتر کپیه امضا نمی‌خواهد | دفتر کپیه به امضا نیاز ندارد | persian_review_law medium |
| 76 | law_test | V2-test-LAW-fa-T3-001 | expected_answer | یا صاحب‌منصبان پارکه یا حاکم محل | یا صاحب‌منصبان پارکه (دادسرا) یا حاکم محل | persian_review_law low; rule 4 |
| 77 | law_test | V2-test-LAW-fa-T3-002 | expected_answer | مهلت در هر دو قانون سه روز | مهلت در هر دو قانون ۳ روز | rule 5 |
| 78 | law_test | V2-test-LAW-fa-T3-002 | expected_answer | را هم به همان دفتر بدهد | را هم به همان دفتر تسلیم کند | persian_review_law medium; rule 1 |
| 79 | law_test | V2-test-LAW-fa-T3-003 | question | یعنی نرخ درآمد مشمول مالیات بیش از ۵۰ میلیون ریال در سال | یعنی نرخ مالیاتِ طبقهٔ درآمدیِ بیش از ۵۰ میلیون ریال در سال | persian_review_law medium |
| 80 | law_test | V2-test-LAW-fa-T3-003 | expected_answer | نرخ درآمد مشمول مالیات «از ۵۰ میلیون ریال به بالا» ۵۵ درصد بود | نرخ مالیاتِ طبقهٔ «از ۵۰ میلیون ریال به بالا» ۵۵ درصد بود | persian_review_law medium |
| 81 | law_test | V2-test-LAW-fa-T3-003 | expected_answer | را عوض کرد و نرخ مازاد بر ۵۰ میلیون ریال را ۶۰ درصد گذاشت | را تغییر داد و نرخ مازاد بر ۵۰ میلیون ریال را ۶۰ درصد تعیین کرد | persian_review_law high |
| 82 | law_test | V2-test-LAW-fa-T3-003 | expected_answer | نرخ‌های جدید دربارهٔ درآمدهایی اعمال می‌شد که از اول سال ۱۳۵۲ به بعد تحصیل شوند، و برای اشخاص حقوقی دربارهٔ سال مالی‌ای که از ۱۳۵۲ به بعد پایان یابد. | نرخ‌های جدید برای درآمدهایی اعمال شد که از آغاز سال ۱۳۵۲ به بعد تحصیل شوند؛ برای اشخاص حقوقی نیز ملاک، سال مالی‌ای بود که از ۱۳۵۲ به بعد پایان یابد. | persian_review_law medium |
| 83 | law_test | V2-test-LAW-fa-T3-004 | question | و خودش در خانهٔ اجاره‌ای دیگری زندگی می‌کند | و خود در خانهٔ دیگری مستأجر است | persian_review_law medium |
| 84 | law_test | V2-test-LAW-fa-T3-004 | question | اجاره‌ای که خودش می‌پردازد در چه صورتی | اجاره‌بهایی که می‌پردازد در چه صورتی | persian_review_law low (accepted) |
| 85 | law_test | V2-test-LAW-fa-T3-004 | expected_answer | و «ملک شخصی محل سکونت خود را» را به «خانه خود را» تغییر داد | و این اصلاحیه عبارت «ملک شخصی محل سکونت خود» را به «خانه خود» تغییر داد | persian_review_law medium |
| 86 | law_test | V2-test-LAW-fa-AB-001 | expected_answer | حداقلی برای سرمایهٔ شرکت سهامی در قانون تجارت ۱۳۰۳ نیامده است | حداقل سرمایه‌ای برای شرکت سهامی در قانون تجارت ۱۳۰۳ نیامده است | persian_review_law medium |
| 87 | law_test | V2-test-LAW-fa-AB-001 | expected_answer | (پنج یا ده تومان) | (۵ یا ۱۰ تومان) | rule 5 |
| 88 | law_test | V2-test-LAW-fa-AB-001 | expected_answer | پرداخت دست‌کم ثلث آن را شرط تشکیل می‌داند | پرداخت دست‌کم یک‌سوم آن را شرط تشکیل شرکت می‌داند | persian_review_law low; rule 4 |
| 89 | law_test | V2-test-LAW-fa-AB-001 | expected_answer | پس عددی برای حداقل سرمایه نمی‌توان داد | پس نمی‌توان رقمی برای حداقل سرمایه ذکر کرد | persian_review_law medium; rule 3 |
| 90 | law_test | V2-test-LAW-fa-T3-005 | question | این دفاتر کدام‌اند و بعد از اصلاحیهٔ ۱۳۵۱ چه دفاتری جای این ارجاع را گرفت؟ | قانون تجارت کدام دفاتر را برای تاجر الزامی کرده است و اصلاحیهٔ ۱۳۵۱ به‌جای این ارجاع کدام دفاتر را نام برد؟ | persian_review_law medium + fidelity (question now asks what the Code requires of a merchant) |
| 91 | law_test | V2-test-LAW-fa-T3-005 | question_gloss_en | Which books are those, and what replaced that reference after the 1351 amendment? | Which books does the Commercial Code require of a merchant, and which books did the 1351 amendment name instead of that reference? | gloss follows question |
| 92 | law_test | V2-test-LAW-fa-T3-005 | expected_answer | قانون تجارت چهار دفتر را برای هر تاجر الزامی می‌کند: دفتر روزنامه، دفتر کل، دفتر دارایی و دفتر کپیه (ماده ۶). | ماده ۶ قانون تجارت چهار دفتر را برای تاجر الزامی می‌کند: دفتر روزنامه، دفتر کل، دفتر دارایی و دفتر کپیه. در قانون تجارت فعلی (۱۳۱۱) کسبهٔ جزء از این تکلیف مستثنا هستند (تشخیص آنان طبق نظام‌نامهٔ وزارت عدلیه است، ماده ۱۹) و تاجر مکلف است این دفاتر یا دفاتری را داشته باشد که وزارت عدلیه با نظام‌نامه قائم‌مقام آن‌ها قرار می‌دهد؛ قانون ۱۳۰۳ این استثنا را نداشت. | fidelity (persian_review_law); Art. 6 1311 checked |
| 93 | law_test | V2-test-LAW-fa-T3-005 | expected_answer_gloss_en | The Commercial Code requires four books: journal, ledger, inventory and copy book (Art. 6). | Art. 6 of the Commercial Code requires a merchant to keep four books: journal, ledger, inventory and copy book. Under the current (1311) Code petty traders are exempt (identified under Ministry of Justice regulations, Art. 19) and a merchant must keep these books or the substitutes the Ministry prescribes by regulation; the 1303 Code had no such exception. | fidelity (gloss) |
| 94 | rpt_test | V2-test-RPT-fa-T1-002 | expected_answer | در کمتر از سه روز رسیدگی شد | در کمتر از ۳ روز رسیدگی شد | rule 5 |
| 95 | rpt_test | V2-test-RPT-fa-T3-004 | expected_answer | دو سال بعد (۱۴۰۴) | ۲ سال بعد (۱۴۰۴) | rule 5 |
| 96 | book_dev | V2-dev-BOOK-fa-T1-002 | expected_answer | در چهار جا: | در ۴ جا: | rule 5 |
| 97 | book_test | V2-test-BOOK-fa-T1-002 | expected_answer | به نوشتهٔ کسروی شش کرور کسری داشت (در جای دیگری از همین کتاب، کسری سالانه شش کرور و نیم آمده است) | به نوشتهٔ کسروی ۶ کرور کسری داشت (در جای دیگری از همین کتاب، کسری سالانه ۶ کرور و نیم آمده است) | rule 5 |
| 98 | book_test | V2-test-BOOK-fa-T1-003 | expected_answer | دارسی باید بیست هزار لیره نقد و همین اندازه سهم شرکت به دولت ایران می‌داد و شش درصد از سود خالص هم از آن ایران بود. مدت امتیاز شصت سال بود | دارسی باید ۲۰ هزار لیره نقد و همین اندازه سهم شرکت به دولت ایران می‌داد و ۶ درصد از سود خالص هم از آن ایران بود. مدت امتیاز ۶۰ سال بود | rule 5 |
| 99 | book_test | V2-test-BOOK-fa-T1-005 | expected_answer | پس از سه ماه و چند روز | پس از ۳ ماه و چند روز | rule 5 |
| 100 | book_test | V2-test-BOOK-fa-T2-001 | expected_answer | هشت سال پادشاه انشان بود | ۸ سال پادشاه انشان بود | rule 5 |
| 101 | book_test | V2-test-BOOK-fa-T2-003 | expected_answer | حدود سه ماه (نزدیک ۹۵ روز) | حدود ۳ ماه (نزدیک ۹۵ روز) | rule 5 |
| 102 | book_test | V2-test-BOOK-fa-T2-004 | expected_answer | در پنج سالی که پس از آن زنده بود | در ۵ سالی که پس از آن زنده بود | rule 5 |
| 103 | book_test | V2-test-BOOK-fa-T2-005 | expected_answer | یا پنجاه یا چهل درصد از سهام خود | یا ۵۰ یا ۴۰ درصد از سهام خود | rule 5 |
| 104 | book_test | V2-test-BOOK-fa-AB-001 | expected_answer | حدود سی هزار کارگر دارد | حدود ۳۰ هزار کارگر دارد | rule 5 |
| 105 | book_test | V2-test-BOOK-fa-AB-001 | expected_answer | یک سال پس از فوران نفتِ ۱۹۰۸ | ۱ سال پس از فوران نفتِ ۱۹۰۸ | rule 5 |
| - | law_test | V2-test-LAW-fa-T3-005 | requirements/evidence | - | + optional R4 (exception for کسبه جزء, substitute books); + E13 Art. 19 (1311) p1_b65 role exception; E2, E8 now also support R4; bundles unchanged | fidelity item |

## Step 2 - independent re-check of the replacement questions
Three independent read-only checkers (one per domain) re-read the sources of the 10 replacement records.
OK as is: RPT fa-T2-003 (Snapp active drivers), en-T2-003 (ArcelorMittal LTIFR); BOOK en-T2-002 (Malthus), fa-T3-005 (Cyrus tomb; cautions: p1_b1471 names Cyrus only by context, 1811 inferred from p1_b697/p1_b700 - both acceptable); LAW fa-T3-003 (55->60%; the amendment's only 55% is another bracket), fa-T2-002 (Guardian Council; Art. 98 three-quarters rule is interpretation, not review), fa-T2-003 (Registration Act; Art. 47 is about unregistered property). All quotes exact, offsets and text hashes match.

| file | id | field | change | why |
|---|---|---|---|---|
| rpt_test | V2-test-RPT-en-T2-001 | evidence | + E4 p7_b17, E5 p51_b11 ('The company expects to harvest 52k GWT of salmon in 2025.') for R3; + bundles [E1,E4], [E1,E5]; restatement_search corrected | re-check: the volume sentence also sits in the CEO letter and p51 |
| rpt_test | V2-test-RPT-en-T2-002 | question | '...return on invested capital did...' -> '...return on invested capital (as reported, not the underlying ROIC) did...' | re-check: underlying ROIC 12.0% is a near-miss |
| book_test | V2-test-BOOK-en-T2-005 | question/answer/requirements | question asks for the loss Veblen concedes; R1 = impairs personal attractions (E2); lower vitality/unfit for work moved to optional R5 (E1); answer: 'plebeian canons' tied to the newly rich only | re-check: p1_b219 alone roughly answered the old 'cost' |

## Step 3a - content_hash / revision recomputed
content_hash = sha256 of canonical JSON (sort_keys, no spaces, UTF-8) over question, expected_answer, answer_format, requirements, evidence, evidence_bundles (SCHEMA.json). Every record whose hash changed relative to *.pre_final.jsonl, or whose stored hash was stale, gets revision = pre-final revision + 1. Gloss-only edits do not change the hash (glosses are outside it) and keep their revision.

63 records re-hashed:

| file | id | revision | why |
|---|---|---|---|
| rpt_dev | V2-dev-RPT-fa-T1-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T1-002 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T1-003 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T2-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T2-002 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T2-003 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T3-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T3-002 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-T3-003 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_dev | V2-dev-RPT-fa-AB-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T1-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T1-002 | 1 -> 2 | edited in this pass + hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T1-004 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T1-005 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T2-001 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T2-002 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T2-003 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T2-004 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T2-005 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T3-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T3-002 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T3-003 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T3-004 | 1 -> 2 | edited in this pass + hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-T3-005 | 2 -> 3 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-fa-AB-001 | 1 -> 2 | hash was stale (earlier Persian pass not re-hashed) |
| rpt_test | V2-test-RPT-en-T2-001 | 1 -> 2 | edited in this pass |
| rpt_test | V2-test-RPT-en-T2-002 | 1 -> 2 | edited in this pass |
| book_dev | V2-dev-BOOK-fa-T1-002 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T1-002 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T1-003 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T1-005 | 1 -> 2 | edited in this pass |
| book_test | V2-test-BOOK-fa-T2-001 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T2-003 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T2-004 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-T2-005 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-fa-AB-001 | 2 -> 3 | edited in this pass |
| book_test | V2-test-BOOK-en-T2-005 | 2 -> 3 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T1-001 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T1-002 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T1-003 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T2-001 | 2 -> 3 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T2-002 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T2-003 | 2 -> 3 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T3-001 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T3-002 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-T3-003 | 1 -> 2 | edited in this pass |
| law_dev | V2-dev-LAW-fa-AB-001 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T1-001 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T1-002 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T1-003 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T1-004 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T1-005 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T2-001 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T2-002 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T2-003 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T2-004 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T2-005 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T3-001 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T3-002 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T3-003 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-T3-004 | 1 -> 2 | edited in this pass |
| law_test | V2-test-LAW-fa-T3-005 | 2 -> 3 | edited in this pass |
| law_test | V2-test-LAW-fa-AB-001 | 2 -> 3 | edited in this pass |

## Step 3b - full self-check (all six files)
Script: scratchpad/final/final_check.py (jsonschema 4.26 installed only into scratchpad/final/pylib, Draft 2020-12 against SCHEMA.json).
Result: 156 records (rpt/book/law: dev 20, test 32 each; per language 3/5 per tier + 1 unanswerable), 0 errors.
Checked: schema per record; ids match split/domain/language, no duplicate ids across the six files; every unit id exists;
every quote an exact substring of its unit and equal to the document text at [start, end); text_sha256; supports and
derivation inputs exist; every bundle supports every required requirement; T1 one doc, T2 one doc with >= 2 regions,
T3 >= 2 docs; T2/T3 audit flags; dev/test disjoint by documents, source blocks and source families in each domain;
content_hash current. The three domain checks (scratchpad/rpt|book|law/check.py) also pass with 0 errors; known
warnings unchanged: rpt en-T3-003 (Rio Tinto table cited elsewhere, different row), book dev-en-T3-003 ('Douglass' named in a comparison question).

## Step 4 - review game ready (review_game/server.py, index.html, start_review.bat; originals in scratchpad/final/game_orig)
- server.py: puts the upgrades worktree (../RAG-Paper-upgrades or %RAG_UPGRADES_DIR%) first on sys.path, so BOOK (K.*) and LAW (L.*) docs load via load_set("BOOK_fa"...); clear error if they cannot.
- question-v2 display: badges tier (or 'unanswerable (intended Tx)') / subtype / domain / language / split / format / rev; EN glosses under question and answer; card "Answer points and audit" (requirements with required/optional, value, unit, derivation; evidence bundles; necessity audit with removal test and restatement search; absence audit with near misses; notes); evidence roles 'E2 · qualifier' with 'supports R2 · region'; documents-needed card from the bundles (T3).
- Backups (*.v1, *.pre_fa, *.pre_final ...) are no longer listed; the six files appear as work/datasets_v2/questions/*.jsonl.
- Test (server hidden on port 8771, then stopped): all six files load; 364 evidence units, 405 quote highlights, all matched exactly at their stored offsets, 0 errors, 0 quotes not found. Browser check: fa records render right-to-left (question, answer, evidence), en left-to-right; audits and glosses shown. One rating was saved only on a temporary copy (scratchpad/final/gametest); no *.review.jsonl exists under work/.
- Start: double-click review_game\start_review.bat (opens http://localhost:8770/), choose a file in the drop-down; ratings go to <file>.review.jsonl next to it.

- 2026-09-25 V2-dev-BOOK-fa-T1-002 (researcher's game note): answer now says Kasravi *knows of* four places (not 'only four') and adds Liqvan (known 60-70 years earlier, since vanished); new evidence E2 = exact quote from Azari p1_b153, new required R5; same paragraph, so it stays T1. Revision 4, content_hash recomputed; self-check 0 errors.
