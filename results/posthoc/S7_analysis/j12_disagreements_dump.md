## 0. HYBRID_6000 V2-test-RPT-en-T2-002 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: What return on invested capital (as reported, not the underlying ROIC) did A.P. Moller - Maersk report for the group for 2024, how did it compare with its yearly target, and how does Maersk calculate this measure?
REF: 12.3% (last 12 months), above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
ANS: A.P. Moller - Maersk reported a group ROIC (last 12 months) of 12.3% for 2024.

- This was above the yearly target of above 7.5% under normalised conditions; profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target (Roadmap towards 2025).
- The average ROIC over 2021-2024 was 31.3%, well above the >12% target for 2021-2025 (Roadmap towards 2025).
- ROIC is defined as profit/loss before financial items for the year (EBIT) less tax on EBIT, divided by the average invested capital, last twelve months (Definition of terms).
REQ: {"R1": "Group ROIC (LTM) was 12.3% in 2024.", "R2": "It was above the yearly target of above 7.5%.", "R3": "ROIC = EBIT less tax on EBIT divided by average invested capital, last twelve months."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target", "unit": null, "status": "unsupported"}, {"text": "The average ROIC over 2021-2024 was 31.3%, well above the >12% target for 2021-2025", "unit": null, "status": "unsupported"}]
strict reason: Two unsupported claims, but all required units correct.
lax reason: All required claims are present and accurate; one unsupported extra claim added.

## 1. HYBRID_6000 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: بودجهٔ سالانهٔ دولت سالانه شش کرور تومان کسری داشت و کمیسیون مجلس دو راه‌حل را در نظر گرفت: یکی افزودن بر درآمد دولت و دیگری کاستن از دریافت‌ها (هزینه‌ها).

- درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌داشت و می‌بایست جای آن را با وام پر کردند (گفتار چهارم).
- کمیسیون مالی که از شش ماه پیش کار را آغاز کرده بود، برای نخستین بار دررفت و درآمد دولت را یکسان گردانید (گفتار هشتم، سید علی‌آقا یزدی).
- کمیسیون برای کاستن از هزینه‌ها، از چهار راه سالانه هشت کرور کم کرد؛ از جمله کاهش مستمری شاهزادگان (مثل شعاع‌السلطنه ۱۱۵ و ظل‌السلطان ۷۵ هزار تومان که هر یک به ۱۲ هزار تومان سالانه نهاده شد) (گفتار هشتم).
- کمیسیون برای این کار برگزیده شده بود و وثوق‌الدوله یکی از اعضای آن بود (گفتار چهارم، سیداکبر شاه).
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌داشت", "unit": "R1", "status": "extra"}, {"text": "می‌بایست جای آن را با وام پر کردند", "unit": null, "status": "unsupported"}, {"text": "کمیسیون مالی که از شش ماه پیش کار را آغاز کرده بود، برای نخستین بار دررفت و درآمد دولت را یکسان گردانید", "unit": null, "status": "unsupported"}, {"text": "کمیسیون برای کاستن از هزینه‌ها، از چهار راه سالانه هشت کرور کم کرد؛ از جمله کاهش مستمری شاهزادگان", "unit": null, "status": "unsupported"}, {"text": "کمیسیون برای این کار برگزیده شده بود و وثوق‌الدوله یکی از اعضای آن بود", "unit": null, "status": "extra"}]
strict reason: Core claims match; extra/unsupported details do not contradict.
lax reason: Main required facts are present and accurate; additional details are non‑essential.

## 2. HYBRID_6000 V2-test-BOOK-fa-T2-001 (T2 fa) stricter=J1 structural=['claim_unsupported']
Q: Pirnia counts Cyrus's reign as 22 years but Herodotus gives 28 and Ctesias 30. What explains the gap, and how long had Cyrus ruled Anshan before taking Hamadan?
REF: 22 years runs from the capture of Hamadan (550 BC) to his death (529 BC); those historians also counted his years of rule before taking Hamadan. Per Nabonidus he had ruled Anshan for eight years before taking Hamadan.
ANS: پیرنیا مدت سلطنت کوروش را از زمان تسخیر همدان تا فوت او در 529 ق. م، یعنی 22 سال می‌داند. علت اختلاف با هرودوت (28 سال) و کتزیاس، دی‌نن و تروگ پومپه (30 سال) این است که آن مورخین سلطنت سابق کوروش را قبل از تسخیر همدان نیز بحساب آورده‌اند. کوروش موافق نوشته نبونید قبل از فتح همدان در مدت هشت سال سلطنت انشان را داشته است. تاریخ فتح همدان 550 ق. م بود و نبونید گفته جنگ کوروش با اژدهاک سه سال طول کشید. (فصل دوم - سلطنت کوروش بزرگ، بخش «مقایسه» و «لوحهٔ نبونید»)
REQ: {"R1": "The 22 years run from the capture of Hamadan to Cyrus's death in 529 BC.", "R2": "The Greek historians counted his earlier reign before Hamadan.", "R3": "Per Nabonidus, Cyrus ruled Anshan for eight years before taking Hamadan."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "تاریخ فتح همدان 550 ق. م بود", "unit": null, "status": "extra"}, {"text": "نبونید گفته جنگ کوروش با اژدهاک سه سال طول کشید", "unit": null, "status": "unsupported"}]
strict reason: All required claims are present and accurate; extra and unsupported statements do not affect required units.
lax reason: All required claims present and correct; extra details harmless.

## 3. HYBRID_6000 V2-test-BOOK-fa-T2-005 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: Per Bahar, under the northern oil concession law, what would happen if Standard Oil transferred the concession to another foreign government or company? And what did later negotiations reveal the company wanted regarding the Southern Oil Company?
REF: Under article 5 the company could not transfer the concession to a foreign government or company; if it did, the concession would be void. Later negotiations showed Standard Oil wanted to bring the Southern Oil Company into the concession or give it a 50% or 40% share of its stock - what the law forbade.
ANS: طبق مادهٔ پنجم امتیازنامه، در صورت تخلف و واگذاری امتیاز به دولت یا شرکت خارجی، امتیاز ملغی (فسخ) می‌شد؛ و در مذاکرات بعدی روشن شد که استاندارد اویل می‌خواهد کمپانی نفت جنوب را در امتیاز دخیل کند یا او را به قرار صد پنجاه یا صد چهل در سهام خود شریک سازد.

- مادهٔ ۵ امتیازنامه (بند ۳ شواهد): کمپانی حق واگذاری امتیاز به دولت یا شرکت خارجی را نداشت و در صورت تخلف امتیاز ملغی می‌شد.
- مادهٔ ۵ همچنین انتقال امتیاز به هر کس و هر کمپانی یا مشارکت دادن دیگری را صریحاً منع و موجب فسخ امتیاز می‌دانست (شواهد ۱ و ۶)، به همین دلیل کابینهٔ مشیرالدوله پیشنهاد مشارکت کمپانی انگلیسی را رد کرد.
- در مذاکرات بعدی (شاهد ۴) معلوم شد کمپانی می‌خواهد کمپانی نفت جنوب را در امتیاز دخیل یا به قرار صد پنجاه یا صد چهل در اسهام خود شرکت دهد.
- این قضیه باعث تحریک احساسات مجلس شد (شاهد ۴).
REQ: {"R1": "Under article 5, transfer to a foreign government or company would void the concession.", "R2": "Standard Oil wanted to bring the Southern Oil Company into the concession or give it a 50% or 40% share.", "R3": "That demand ran against the law's transfer ban."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "مادهٔ ۵ امتیازنامه انتقال امتیاز به هر کس و هر کمپانی یا مشارکت دادن دیگری را صریحاً منع و موجب فسخ امتیاز می‌دانست", "unit": "R1", "status": "extra"}, {"text": "کابینهٔ مشیرالدوله پیشنهاد مشارکت کمپانی انگلیسی را رد کرد", "unit": null, "status": "unsupported"}, {"text": "این قضیه باعث تحریک احساسات مجلس شد", "unit": null, "status": "unsupported"}]
strict reason: R3 missing; extra and unsupported claims present.
lax reason: All required facts are present and accurate; no contradictions; extra details do not affect answer.

## 4. HYBRID_6000 V2-test-BOOK-en-T3-001 (T3 en) stricter=J2 structural=['claim_wrong_value']
Q: Darwin estimates how fast the slow-breeding elephant would multiply from one pair. How did his assumptions and his result change between the first edition (1859) and the sixth edition (1872) of the Origin of Species?
REF: First edition: breeding from thirty to ninety years of age with three pairs of young, giving fifteen million elephants at the end of the fifth century (500 years). Sixth edition: breeding from thirty to ninety, six young in that interval and living to one hundred, giving nearly nineteen million after 740 to 750 years.
ANS: 1st edition (1859) - elephant estimate: breeds at thirty years old, breeds till ninety, bringing forth three pair of young; at the end of the fifth century there would be alive fifteen million elephants descended from the first pair.
6th edition (1872) - elephant estimate: begins breeding when thirty years old, breeds till ninety, bringing forth six young in the interval, surviving till one hundred years old; after a period of from 740 to 750 years there would be nearly nineteen million elephants descended from the first pair.

Between editions, Darwin added the assumption that elephants survive till one hundred years old and changed the offspring from "three pair" (six young) to "six young" (three per pair); the result changed from fifteen million elephants at the end of the fifth century to nearly nineteen million elephants after 740 to 750 years.
REQ: {"R1": "1st ed: three pairs of young between 30 and 90.", "R2": "1st ed: fifteen million after the fifth century (~500 years).", "R3": "6th ed: six young between 30 and 90, surviving to 100.", "R4": "6th ed: nearly nineteen million after 740-750 years."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): [{"text": "Darwin added the assumption that elephants survive till one hundred years old", "unit": null, "status": "extra"}, {"text": "Changed the offspring from 'three pair' (six young) to 'six young' (three per pair)", "unit": null, "status": "wrong_value"}]
strict reason: All required facts correct; one extra and one misinterpreted offspring detail.
lax reason: All required facts are accurately presented; extra comparative note is harmless.

## 5. HR_6000 V2-test-RPT-fa-T2-003 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: How many active drivers did Snapp's car service have in 1403, and whom does Snapp count as an 'active' driver?
REF: 3,692,657 active driver users, about half of the 7,417,231 driver users in total. An active driver is one with at least one trip on Snapp car services in 1403.
ANS: اسنپ در سال ۱۴۰۳ تعداد کاربران رانندهٔ فعال را ۳,۶۹۲,۶۵۷ نفر گزارش کرده است [6].

طبق واژه‌نامهٔ گزارش ۱۴۰۳، «فعال» به کاربری گفته می‌شود که در آن سال حداقل ۱ سفر یا سفارش از سرویس نامبرده داشته باشد [4]. همچنین اسنپ رانندگان را به دو دسته تقسیم می‌کند: رانندهٔ تمام‌وقت کسی است که بیش از ۴۰ سفر موفق در ماه انجام دهد و رانندهٔ پاره‌وقت کسی که کمتر از ۴۰ سفر موفق در ماه انجام دهد [4].
REQ: {"R1": "Snapp had 3,692,657 active driver users in 1403.", "R2": "An active driver is a user with at least one trip on Snapp car services in 1403.", "R3": "Snapp had 7,417,231 driver users in total (about half were active)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "رانندگان به دو دسته تقسیم می‌شوند: رانندهٔ تمام‌وقت کسی است که بیش از ۴۰ سفر موفق در ماه انجام دهد", "unit": null, "status": "unsupported"}, {"text": "رانندهٔ پاره‌وقت کسی که کمتر از ۴۰ سفر موفق در ماه انجام دهد", "unit": null, "status": "unsupported"}, {"text": "[6]", "unit": null, "status": "non_substantive"}, {"text": "[4]", "unit": null, "status": "non_substantive"}]
strict reason: R1 and R2 correct; R3 missing; extra driver categorization unsupported.
lax reason: All required claims present and accurate; extra detail does not affect answer.

## 6. HR_6000 V2-test-RPT-fa-T3-001 (T3 fa) stricter=J2 structural=['claim_wrong_value']
Q: How much did the total distance travelled by the Digi Express fleet change in 1403 compared with 1402?
REF: From 50 million km in 1402 to 54.6 million km in 1403: +4.6 million km (about 9%).
ANS: دیجی‌اکسپرس - 1402: 50 میلیون کیلومتر
دیجی‌اکسپرس - 1403: 54.6 میلیون کیلومتر

مجموع مسافت در سال 1403 نسبت به 1402 حدود 4.6 میلیون کیلومتر (حدود 9.2٪) افزایش یافته است.
REQ: {"R1": "Digi Express vehicles travelled 50 million km in 1402.", "R2": "Digi Express fleet travelled 54.6 million km in 1403.", "R3": "The increase was 4.6 million km (about 9%)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "حدود 9.2٪ افزایش یافته است", "unit": "R3", "status": "wrong_value"}]
strict reason: R3 value mismatch (9% vs 9.2%); derived claim partially correct.
lax reason: All claims match required facts; derived claim value correct, extra percentage detail acceptable.

## 7. HR_6000 V2-test-RPT-en-T2-002 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: What return on invested capital (as reported, not the underlying ROIC) did A.P. Moller - Maersk report for the group for 2024, how did it compare with its yearly target, and how does Maersk calculate this measure?
REF: 12.3% (last 12 months), above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
ANS: A.P. Moller - Maersk reported a group ROIC (LTM) of 12.3% for 2024, which was above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items for the year (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.

- 2024 ROIC of 12.3% (up from 7.4% in 2023), above the >7.5% yearly target; profitability lagged in the first half of 2024, but the strong second half brought ROIC above target (Roadmap towards 2025; Financial review – Highlights for the year).
- Definition: EBIT less tax on EBIT divided by average invested capital, last twelve months (Definition of terms).
REQ: {"R1": "Group ROIC (LTM) was 12.3% in 2024.", "R2": "It was above the yearly target of above 7.5%.", "R3": "ROIC = EBIT less tax on EBIT divided by average invested capital, last twelve months."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "2024 ROIC of 12.3% (up from 7.4% in 2023)", "unit": null, "status": "extra"}, {"text": "profitability lagged in the first half of 2024, but the strong second half brought ROIC above target", "unit": null, "status": "unsupported"}, {"text": "Definition: EBIT less tax on EBIT divided by average invested capital, last twelve months", "unit": "R3", "status": "duplicate"}]
strict reason: All required claims correct; extra and unsupported details present but harmless.
lax reason: All required claims match reference; extra details do not affect correctness.

## 8. HR_6000 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: بودجهٔ سالانهٔ دولت شش کرور (و نیم) کسری داشت و کمیسیون دو راه در نظر گرفت: یکی افزودن به درآمد و دیگری کاستن از دریافت‌ها.

- درآمد دولت سالانه پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌آورد و می‌بایست جای آن با وام پر گردد (گفتار چهارم، بندهای [4] و [5]).
- کمیسیون مالی که وثوق‌الدوله یکی از اعضای آن بود، دو کار را اندیشه گرفت: افزودن به درآمد و کاستن از دریافت‌ها (گفتار چهارم، بند [1] و [3]).
- برای افزونی درآمد، از جمله «تفاوت عمل» به حساب دولت درآمد و تیول برانداخته شد، و برای کاستن از دریافت‌ها، از مستمری شاهزادگان و دیگران کاسته شد (گفتار چهارم، بندهای [3] و [7]، و گفتار هشتم، بند [2]).
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد دولت سالانه پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌آورد", "unit": "R1", "status": "extra"}, {"text": "می‌بایست جای آن با وام پر گردد", "unit": null, "status": "unsupported"}, {"text": "کمیسیون مالی که وثوق‌الدوله یکی از اعضای آن بود", "unit": null, "status": "extra"}, {"text": "برای افزونی درآمد، از جمله «تفاوت عمل» به حساب دولت درآمد و تیول برانداخته شد", "unit": null, "status": "unsupported"}, {"text": "برای کاستن از دریافت‌ها، از مستمری شاهزادگان و دیگران کاسته شد", "unit": null, "status": "unsupported"}, {"text": "(گفتار چهارم، بندهای [4] و [5])", "unit": null, "status": "non_substantive"}, {"text": "(گفتار چهارم، بند [1] و [3])", "unit": null, "status": "non_substantive"}, {"text": "(گفتار چهارم، بندهای [3] و [7]، و گفتار هشتم، بند [2])", "unit": null, "status": "non_substantive"}]
strict reason: Core claims correct; extra/unsupported details do not contradict.
lax reason: Core claims match required facts; additional details are non‑essential extras.

## 9. HR_6000 V2-test-BOOK-fa-T3-002 (T3 fa) stricter=J2 structural=['claim_unsupported']
Q: The man elected first vice-president of the first Majlis: what was his role in the 1919 agreement, and per Britain's later claim, what sum had been shared between him and two other ministers?
REF: Vosough al-Dowleh. He concluded the 1919 agreement with Britain (subject to Majlis approval). Per Britain's claim, finance minister Sarem al-Dowleh received £130,000 at the signing, and this sum was shared between Vosough al-Dowleh, Sarem al-Dowleh and Nosrat al-Dowleh.
ANS: پاسخ: وثوق‌الدوله.

- او در قرارداد ۱۹۱۹ (بسته‌شده در ذی‌القعده ۱۳۳۷ / دسامبر ۱۹۱۹ میان دولت ایران و دولت بریتانیا) طرف ایرانی قرارداد بود و در سال دوم حکومت خود با تقاضاهای جدیدِ زادهٔ انقلاب روسیه مواجه شد که به‌صورت قرارداد به دولت ایران پیشنهاد گردید (بخش «قرارداد ۱۹۱۹ و احمدشاه»).
- طبق مطالبهٔ دولت بریتانیا پس از قضایای انحلال پلیس جنوب، مبلغ صد و سی هزار لیره که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ شده بود، بین وثوق‌الدوله و صارم‌الدوله و نصرةالدوله تقسیم شده بود (بخش «آغاز ضعف دولت قوام»).
REQ: {"R1": "The first vice-president of the first Majlis was Vosough al-Dowleh.", "R2": "He concluded the 1919 agreement with Britain.", "R3": "£130,000, received by Sarem al-Dowleh, was shared between Vosough, Sarem and Nosrat al-Dowleh."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "قرارداد ۱۹۱۹ در ذی‌القعده ۱۳۳۷ / دسامبر ۱۹۱۹ بسته شد.", "unit": null, "status": "extra"}, {"text": "در سال دوم حکومت وثوق‌الدوله، تقاضاهای جدید زادهٔ انقلاب روسیه به دولت ایران پیشنهاد گردید.", "unit": null, "status": "unsupported"}, {"text": "مطالبهٔ دولت بریتانیا پس از قضایای انحلال پلیس جنوب مطرح شد.", "unit": null, "status": "unsupported"}]
strict reason: All required claims correct; extra and unsupported details present.
lax reason: All three required claims are present and accurately reflect the reference.

## 10. HR_6000 V2-test-BOOK-en-T3-005 (T3 en) stricter=J1 structural=['claim_unsupported', 'req_missing']
Q: What does Mill say majority feeling in the United States does to people with large incomes who would like to live showily, and how does Veblen's theory explain the motive that this feeling restrains?
REF: Mill: in the most democratic country, the majority's dislike of any more showy or costly style of living than they can rival works as a tolerably effectual sumptuary law, so that a person with a very large income finds it hard to spend it in any way that will not incur popular disapproval (he thinks the reports exaggerated but the result probable). Veblen: the end of accumulation is to rank high in pecuniary strength compared with the rest of the community, and conspicuous consumption of valuable goods is a means of reputability for the gentleman of leisure; showy spending is how wealth earns esteem.
ANS: [1] On Liberty, Chapter IV - Mill's claim: the feeling of the majority in the United States, to whom any appearance of a more showy or costly style of living than they can hope to rival is disagreeable, operates as a tolerably effectual sumptuary law, and in many parts of the Union it is really difficult for a person with a very large income to find any mode of spending it that will not incur popular disapprobation.

[4] The Theory of the Leisure Class, Chapter Five - Veblen's theory: the accepted standard of expenditure in the community or class largely determines a person's standard of living, directly by commending itself to his common sense as right and good, and indirectly through popular insistence on conformity to the accepted scale of expenditure as a matter of propriety, under pain of disesteem and ostracism.

Mill says majority feeling in the United States acts as a "tolerably effectual sumptuary law," making it difficult for people with very large incomes to spend money showily without incurring popular disapprobation. Veblen's theory explains the motive being restrained as the drive toward conspicuous, costly expenditure beyond the accepted community standard, which society polices through disesteem and ostracism.
REQ: {"R1": "Mill: majority dislike of showy living works as a sumptuary law, making it hard for the rich to spend without popular disapproval.", "R2": "Veblen: conspicuous consumption of valuable goods is a means of reputability.", "R3": "Veblen: the aim is to rank high in pecuniary strength compared with others."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "Veblen: the accepted standard of expenditure determines a person's standard of living and is enforced by disesteem and ostracism.", "unit": null, "status": "unsupported"}]
strict reason: Mill claim matches; Veblen claim does not state required reputability conclusion; required Veblen claim absent.
lax reason: R1 and R2 correct; R3 missing but optional; extra Veblen detail harmless.

## 11. HR_6000 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: 1. مجموع نمایندگان اقلیت‌های دینی: پنج نفر.
2. زرتشتیان: یک نماینده.
3. کلیمیان: یک نماینده.
4. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده.
5. مسیحیان ارمنی جنوب: یک نماینده.
6. مسیحیان ارمنی شمال: یک نماینده.
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'missing', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده.", "unit": null, "status": "extra"}, {"text": "مسیحیان ارمنی جنوب: یک نماینده.", "unit": null, "status": "extra"}, {"text": "مسیحیان ارمنی شمال: یک نماینده.", "unit": null, "status": "extra"}]
strict reason: All required claims present and correct; missing Jews and total Armenian seats; extra details do not affect correctness.
lax reason: All required claims present and accurate.

## 12. HR_6000 V2-test-LAW-fa-T2-004 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: A barber's permanent licence has expired; the union sent a notice but the owner did not renew, and he is advertising on Instagram. Under the Guild System Law as amended in 1392, what is the shop's status and what fine does the advertising carry?
REF: After the one-month renewal notice, a unit whose licence is not renewed is treated as unlicensed (Art. 5 note 2). Advertising in any way by a guild member without a valid licence is prohibited, with a fine of 1 million to 250 million rials (Art. 17 note 3).
ANS: پاسخ مستقیم: واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود و چون فاقد پروانه کسب معتبر است، انتشار آگهی تبلیغاتی برایش ممنوع و مشمول جریمه نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال است.

مبانی:
1. طبق تبصره 2 ماده 5، مدت اعتبار پروانه کسب دائم پنج‌سال است و اتحادیه مکلف است با انقضای مدت اعتبار، اخطاریه یک‌ماهه برای تمدید صادر کند.
2. طبق همان تبصره 2 ماده 5، در صورت عدم تبدیل یا تمدید پروانه، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود.
3. طبق تبصره 3 ماده 17، انتشار هرگونه آگهی تبلیغاتی به هر طریق توسط فرد صنفی فاقد پروانه کسب معتبر ممنوع است.
4. طبق همان تبصره 3 ماده 17، متخلف به جریمه نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال محکوم می‌شود.
REQ: {"R1": "A unit whose licence is not renewed after the one-month notice is treated as unlicensed.", "R2": "Advertising by a guild member without a valid licence is prohibited.", "R3": "The fine is 1,000,000 to 250,000,000 rials."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "مدت اعتبار پروانه کسب دائم پنج‌سال است", "unit": null, "status": "unsupported"}, {"text": "اتحادیه مکلف است با انقضای مدت اعتبار، اخطاریه یک‌ماهه برای تمدید صادر کند", "unit": "R1", "status": "extra"}]
strict reason: All required claims correct; extra unsupported detail on licence duration.
lax reason: All claims match required facts precisely.

## 13. HR_6000 V2-test-LAW-en-T3-001 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: Both the Digital Services Act and the Digital Markets Act use a 45 million user figure. What exactly does each Act count, and what else does the DMA require alongside that figure?
REF: DSA: 45 million or more average monthly active recipients of the service in the Union is the threshold for very large online platforms and search engines (Art. 33(1)). DMA: 45 million monthly active end users established or located in the Union in the last financial year, together with at least 10,000 yearly active business users established in the Union, triggers the presumption that a core platform service is an important gateway (Art. 3(2)(b)).
ANS: Digital Services Act - threshold: average monthly active recipients of the service in the Union equal to or higher than 45 million (equivalent to 10 % of the Union population), calculated as an average over a period of six months.

Digital Markets Act - threshold: at least 45 million monthly active end users established or located in the Union in the last financial year.

Digital Markets Act - additional requirement: at least 10 000 yearly active business users established in the Union, and the thresholds in point (b) must have been met in each of the last three financial years.

The DSA counts average monthly active recipients of an online platform or search engine service in the Union, while the DMA counts monthly active end users established or located in the Union; alongside the 45 million end users, the DMA also requires at least 10 000 yearly active business users established in the Union, with those thresholds met in each of the last three financial years.
REQ: {"R1": "DSA counts average monthly active recipients of the service in the Union (>= 45 million).", "R2": "DMA counts monthly active end users established or located in the Union in the last financial year.", "R3": "DMA also requires at least 10,000 yearly active business users established in the Union.", "R4": "The DSA figure designates very large online platforms/search engines; the DMA figure triggers the gatekeeper 'important gateway' presumption."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "DSA calculation: average over a period of six months", "unit": null, "status": "extra"}, {"text": "DSA threshold equivalent to 10 % of the Union population", "unit": null, "status": "extra"}, {"text": "DMA thresholds must have been met in each of the last three financial years", "unit": null, "status": "unsupported"}, {"text": "DSA counts average monthly active recipients of an online platform or search engine service in the Union", "unit": "R1", "status": "duplicate"}, {"text": "DMA counts monthly active end users established or located in the Union", "unit": "R2", "status": "duplicate"}, {"text": "DMA also requires at least 10 000 yearly active business users established in the Union", "unit": "R3", "status": "duplicate"}]
strict reason: DMA three-year requirement unsupported; R4 legal consequence missing.
lax reason: Core claims match required facts; extra details are harmless; one duplicate and one extra claim noted.

## 14. HR_6000 V2-test-LAW-en-T3-004 (T3 en) stricter=J2 structural=['claim_unsupported', 'claim_wrong_value', 'req_wrong']
Q: A UK police force (not an undertaking) processes personal data for law-enforcement purposes in breach of the Data Protection Act 2018's first data protection principle (processing must be lawful and fair). An EU company breaches the GDPR's basic principles for processing. What is the maximum fine each could face?
REF: Police force: breach of s.35 (the first data protection principle under Part 3) attracts the 'higher maximum amount' (s.157(2)(a)), which for a body that is not an undertaking is £17,500,000 (s.157(5)(b)). EU company: infringing the basic principles (Arts. 5, 6, 7 and 9) is fined up to EUR 20,000,000 or up to 4 % of total worldwide annual turnover of the preceding financial year, whichever is higher (GDPR Art. 83(5)(a)).
ANS: Part 3 (law enforcement), section 35 breach - Data Protection Act 2018: failure to comply with section 35 attracts the higher maximum amount, which per section 157(5) is £17,500,000 (see evidence [5]).
EU company - GDPR basic principles breach: maximum penalty of 20 million Euros (per evidence [5], substituting for £17,500,000 in Article 83(5)/section 157(5)).

The police force faces the higher maximum amount of £17,500,000, while the EU company faces up to 20 million Euros; the two figures are of similar magnitude but denominated in different currencies, so they are not directly comparable without exchange-rate conversion.
REQ: {"R1": "The police force's breach of s.35 attracts the higher maximum amount.", "R2": "For a non-undertaking the higher maximum is £17,500,000.", "R3": "The GDPR maximum is EUR 20,000,000 or 4% of worldwide annual turnover, whichever is higher."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'wrong'} | lax units: {'R1': 'missing', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "EU company - GDPR basic principles breach: maximum penalty of 20 million Euros", "unit": "R3", "status": "wrong_value"}, {"text": "the two figures are of similar magnitude but denominated in different currencies, so they are not directly comparable without exchange-rate conversion", "unit": null, "status": "extra"}, {"text": "see evidence [5]", "unit": null, "status": "non_substantive"}, {"text": "substituting for £17,500,000 in Article 83(5)/section 157(5)", "unit": null, "status": "unsupported"}]
strict reason: GDPR max fine omits 4% turnover alternative; unsupported substitution claim.
lax reason: Claims match required facts; optional claim omitted; extra commentary does not affect answer.

## 15. LEX5B_6000 V2-test-RPT-en-T2-002 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: What return on invested capital (as reported, not the underlying ROIC) did A.P. Moller - Maersk report for the group for 2024, how did it compare with its yearly target, and how does Maersk calculate this measure?
REF: 12.3% (last 12 months), above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
ANS: A.P. Moller - Maersk reported a group ROIC of 12.3% for 2024, which was above its yearly target of above 7.5% under normalised conditions. Profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target. The measure is defined as profit/loss before financial items for the year (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
REQ: {"R1": "Group ROIC (LTM) was 12.3% in 2024.", "R2": "It was above the yearly target of above 7.5%.", "R3": "ROIC = EBIT less tax on EBIT divided by average invested capital, last twelve months."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "Profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target", "unit": null, "status": "unsupported"}]
strict reason: Two claims correct, one unsupported detail, one correct definition.
lax reason: All required claims are present, accurate, and match the reference.

## 16. LEX5B_6000 V2-test-RPT-en-T2-005 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: By how much did Lerøy's wild-catch volume fall in 2024, and what does the company say about the outlook for that business?
REF: Catch volume fell from 75,893 tonnes (HOG) in 2023 to 64,991 tonnes in 2024 - about 10,900 tonnes or 14% less. Lerøy says WildCatch earnings have been significantly affected by the low quota basis and expects 2025 and probably 2026 to be challenging.
ANS: Catch volume fell from 75 893 tonnes in 2023 to 64 991 tonnes in 2024 (a decrease of 10 902 tonnes), reflecting quota cuts of 34% for cod, 43% for haddock and 6% for saithe in the northern zone [1][2]. The Board says earnings in the WildCatch segment have been significantly affected by the low quota basis and will remain so for some time, expecting 2025 and probably also 2026 to be challenging for the segment [3]. The quota reduction is to some extent offset by higher prices, though these challenge the land-based industry, which made a larger loss in 2024 than 2023 [2].
REQ: {"R1": "Catch volume was 64,991 tonnes (HOG) in 2024 versus 75,893 in 2023.", "R2": "The fall was about 10,900 tonnes (about 14%).", "R3": "Earnings are hit by the low quota basis and 2025 (probably 2026) are expected to be challenging."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "reflecting quota cuts of 34% for cod, 43% for haddock and 6% for saithe in the northern zone", "unit": null, "status": "extra"}, {"text": "The quota reduction is to some extent offset by higher prices, though these challenge the land-based industry, which made a larger loss in 2024 than 2023", "unit": null, "status": "unsupported"}, {"text": "[1][2]", "unit": null, "status": "non_substantive"}, {"text": "[3]", "unit": null, "status": "non_substantive"}]
strict reason: Core claims match requirements; extra/unsupported details do not contradict.
lax reason: All required claims are present and accurate; extra detail does not affect correctness.

## 17. LEX5B_6000 V2-test-RPT-en-T3-003 (T3 en) stricter=J1 structural=['req_missing']
Q: Which employed more people on average during 2024, Rio Tinto or A.P. Moller - Maersk, and by roughly how many?
REF: Maersk: an average of 106,626 employees (headcount; its FTE-based average in note 2.2 is 104,890) versus Rio Tinto's average of 59,594 (FTE basis, including its share of equity-accounted units) - roughly 45,000-51,000 more, about 1.8-1.9 times as many.
ANS: A.P. Moller - Maersk - 2024: 106,626 average employees
Rio Tinto - 2024: 59,594 average employees

A.P. Moller - Maersk employed more people on average during 2024, by roughly 47,032 (106,626 − 59,594).
REQ: {"R1": "Maersk's average number of employees in 2024 was 106,626.", "R2": "Rio Tinto's average number of employees in 2024 was 59,594.", "R3": "Maersk employed roughly 45,000-51,000 more on average (about 1.8-1.9 times).", "R4": "Maersk employed more people than Rio Tinto."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): []
strict reason: All factual claims match; conclusion R4 not explicitly stated.
lax reason: All required claims present and accurate; difference calculation matches reference.

## 18. LEX5B_6000 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: بودجهٔ سالانهٔ دولت سالانه شش کرور تومان (و نیم) کسری داشت و کمیسیون دو راه‌حل را در نظر گرفت: یکی آنکه به درآمد دولت بیفزایند و دیگری آنکه از دریافت‌ها بکاهند.

- درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کسری می‌داشت و می‌بایست جای آن با وام پر شود (گفتار چهارم).
- مجلس برای این کار کمیسیونی برگزید که وثوق‌الدوله یکی از باشندگان آن بود (گفتار چهارم).
- کمیسیون دو کار را اندیشید: افزودن به درآمد و کاستن از دریافت‌ها (گفتار چهارم).
- برای افزونی درآمد، از جمله مقرر شد «تفاوت عمل» (فزونی مالیات‌های حکمرانان) به حساب دولت درآید (گفتار چهارم).
- برای کاستن از دریافت‌ها نیز از مستمری‌های گزاف شاهزادگان کاسته شد و به هر یک ۱۲ هزار تومان سالانه مقرر گشت (گفتار هشتم).
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کسری می‌داشت", "unit": "R1", "status": "extra"}, {"text": "می‌بایست جای آن با وام پر شود", "unit": null, "status": "unsupported"}, {"text": "مجلس برای این کار کمیسیونی برگزید که وثوق‌الدوله یکی از باشندگان آن بود", "unit": null, "status": "extra"}, {"text": "کمیسیون دو کار را اندیشید: افزودن به درآمد و کاستن از دریافت‌ها", "unit": "R2", "status": "duplicate"}, {"text": "برای افزونی درآمد، از جمله مقرر شد «تفاوت عمل» به حساب دولت درآید", "unit": null, "status": "extra"}, {"text": "برای کاستن از دریافت‌ها نیز از مستمری‌های گزاف شاهزادگان کاسته شد و به هر یک ۱۲ هزار تومان سالانه مقرر گشت", "unit": null, "status": "unsupported"}]
strict reason: Core claims match requirements; extra/unsupported details do not contradict.
lax reason: All required claims are present and accurate; other statements are extra details.

## 19. LEX5B_6000 V2-test-BOOK-fa-T1-003 (T1 fa) stricter=J2 structural=['req_missing']
Q: When and from which shah did D'Arcy obtain the oil concession, what did Iran receive, and for how long?
REF: 1319 AH (1901) from Mozaffar al-Din Shah; £20,000 cash plus the same in company shares and 6% of net profits; sixty years, covering all of Iran except Azarbaijan, Gilan, Mazandaran, Astarabad and Khorasan.
ANS: دارسی در سال ۱۳۱۹ (۱۹۰۱ میلادی) امتیاز برآوردن نفت را از مظفرالدین شاه گرفت.

- سال و شاه: در سال ۱۳۱۹ (۱۹۰۱ میلادی) مستر دارسی انگلیسی امتیاز برآوردن نفت در سراسر ایران را از مظفرالدین شاه گرفت (بخش «کاوش در ویرانه‌های شوش» از تاریخ پانصد ساله خوزستان).
- قلمرو امتیاز: سراسر ایران به‌جز آذربایجان و گیلان و مازندران و استرآباد و خراسان.
- عوض ایران: بیست هزار لیره نقد پرداخت، به همان اندازه سهم شرکت برای دولت ایران می‌سپارد و گذشته از آنها صدی شش از نفع خرج دررفته بهره ایران بود.
- مدت امتیاز: شصت سال.
REQ: {"R1": "Granted in 1319 AH (1901 AD).", "R2": "Granted by Mozaffar al-Din Shah.", "R3": "Iran got £20,000 cash, the same amount in shares, and 6% of net profits.", "R4": "The concession ran for sixty years."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): [{"text": "سال و شاه: در سال ۱۳۱۹ (۱۹۰۱ میلادی) مستر دارسی انگلیسی امتیاز برآوردن نفت در سراسر ایران را از مظفرالدین شاه گرفت.", "unit": "R1", "status": "duplicate"}, {"text": "قلمرو امتیاز: سراسر ایران به‌جز آذربایجان و گیلان و مازندران و استرآباد و خراسان.", "unit": null, "status": "extra"}]
strict reason: Missing grantor (R2); extra territory detail; otherwise correct.
lax reason: All required facts are present and accurate; extra territorial detail does not affect correctness.

## 20. LEX5B_6000 V2-test-BOOK-fa-T1-005 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: When did Seyyed Zia's government fall and where did he go afterwards?
REF: On 4 Jowza 1300, after three months and some days; he went via Basra to Europe and stayed in Montreux, Switzerland.
ANS: دولت سیدضیاءالدین بعد از سه ماه و چند روز در 4 جوزا 1300 سقوط کرد و او تا سرحد مشایعت شد و از آنجا از راه بصره به اروپا رفت و در «منترو» از شهرهای سوئیس متوقف گشت (بند [1]). دستخط عزل او نیز در همان روز 4 جوزا 1300 به عموم ایالات و حکام مخابره شد (بند [5]).
REQ: {"R1": "The government fell on 4 Jowza 1300.", "R2": "It lasted three months and some days.", "R3": "He went via Basra to Europe and stayed in Montreux, Switzerland."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "او تا سرحد مشایعت شد", "unit": null, "status": "extra"}, {"text": "دستخط عزل او نیز در همان روز 4 جوزا 1300 به عموم ایالات و حکام مخابره شد", "unit": null, "status": "unsupported"}, {"text": "(بند [1])", "unit": null, "status": "non_substantive"}, {"text": "(بند [5])", "unit": null, "status": "non_substantive"}]
strict reason: R1 and R3 correct; R2 missing; unsupported detail about dismissal decree.
lax reason: All required and optional claims are accurately stated; extra detail does not affect correctness.

## 21. LEX5B_6000 V2-test-BOOK-fa-T2-001 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: Pirnia counts Cyrus's reign as 22 years but Herodotus gives 28 and Ctesias 30. What explains the gap, and how long had Cyrus ruled Anshan before taking Hamadan?
REF: 22 years runs from the capture of Hamadan (550 BC) to his death (529 BC); those historians also counted his years of rule before taking Hamadan. Per Nabonidus he had ruled Anshan for eight years before taking Hamadan.
ANS: علت اختلاف این است که هرودوت و کتزیاس سلطنتِ پیشین کوروش را، یعنی دوره‌ای را که پیش از تسخیر همدان داشت، نیز در حساب سلطنت او آورده‌اند، در حالی که پیرنیا سلطنت را از زمان فتح همدان (۵۵۰ ق. م) تا فوت او (۵۲۹ ق. م) یعنی ۲۲ سال می‌داند. کوروش موافق نوشتهٔ نبونید پیش از فتح همدان هشت سال پادشاه انشان بود.

- پیرنیا فوت کوروش را ۵۲۹ ق. م می‌داند و چون نبونید نوشته کوروش در سال ششم سلطنتش همدان را گرفت، مدت سلطنت از تسخیر همدان ۲۲ سال می‌شود (بند «مقایسه»، فصل دوم).
- هرودوت ۲۸ و کتزیاس و دی‌نن و تروگ پومپه ۳۰ سال نوشته‌اند، زیرا سلطنت سابق او را قبل از تسخیر همدان بحساب آورده‌اند (بند «مقایسه»، فصل دوم).
- موافق نبونید، کوروش قبل از فتح همدان در مدت هشت سال سلطنت انشان را داشته (بخش «لوحهٔ نبونید»، فصل دوم).
REQ: {"R1": "The 22 years run from the capture of Hamadan to Cyrus's death in 529 BC.", "R2": "The Greek historians counted his earlier reign before Hamadan.", "R3": "Per Nabonidus, Cyrus ruled Anshan for eight years before taking Hamadan."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "پیرنیا فوت کوروش را ۵۲۹ ق. م می‌داند.", "unit": "R1", "status": "extra"}, {"text": "نبونید نوشته کوروش در سال ششم سلطنتش همدان را گرفت.", "unit": null, "status": "unsupported"}, {"text": "هرودوت ۲۸ و کتزیاس و دی‌نن و تروگ پومپه ۳۰ سال نوشته‌اند، زیرا سلطنت سابق او را قبل از تسخیر همدان بحساب آورده‌اند.", "unit": "R2", "status": "duplicate"}, {"text": "موافق نبونید، کوروش قبل از فتح همدان در مدت هشت سال سلطنت انشان را داشته.", "unit": "R3", "status": "duplicate"}]
strict reason: All required claims present and correct; extra/unsupported details do not alter core answer.
lax reason: All required claims are present and accurately reflect the reference packet.

## 22. LEX5B_6000 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: طبق اصل ۶۴ قانون اساسی، اقلیت‌های دینی در مجموع ۵ نماینده دارند:

1. زرتشتیان: یک نماینده
2. کلیمیان: یک نماینده
3. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
4. مسیحیان ارمنی جنوب: یک نماینده
5. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}]
strict reason: All required claims present except Jews (R2); extra claim about کلیمیان does not affect answer.
lax reason: All required claims match the reference exactly.

## 23. HYBRID_15500 V2-test-RPT-fa-T3-004 (T3 fa) stricter=J2 structural=['claim_unsupported']
Q: When Snapp Pro launched in 1402, how many services did it cover, and two years later in how many sections and cities was it offered?
REF: At launch in 1402: 3 services (car, food and supermarket with express delivery); two years later (1404): offered in 29 cities and 4 sections (car, food, supermarket with express delivery, and 'doctor and counsellor').
ANS: اسنپ‌پرو - 1402 (راه‌اندازی): فعال روی 3 سرویس خودرو، غذا و سوپرمارکت (تحویل فوری)، در شهرهای تهران، مشهد، اصفهان، شیراز، کرج، قم، اهواز و تبریز
اسنپ‌پرو - 1404: تخفیف‌های ویژه در 4 بخش (خودرو، غذا، سوپرمارکت تحویل فوری و پزشک و مشاور) در 29 شهر

در راه‌اندازی، اسنپ‌پرو 3 سرویس را پوشش می‌داد، اما دو سال بعد در 1404 به 4 بخش گسترش یافت و از 8 شهر به 29 شهر رسید.
REQ: {"R1": "At launch in 1402 Snapp Pro covered 3 services: car, food and supermarket (express).", "R2": "Two years after launch it was offered in 29 cities.", "R3": "Two years after launch it covered 4 sections (car, food, express supermarket, doctor and counsellor)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "اسنپ‌پرو در راه‌اندازی در شهرهای تهران، مشهد، اصفهان، شیراز، کرج، قم، اهواز و تبریز فعال بود", "unit": null, "status": "extra"}, {"text": "در راه‌اندازی، اسنپ‌پرو ۳ سرویس را پوشش می‌داد", "unit": "R1", "status": "duplicate"}, {"text": "دو سال بعد در ۱۴۰۴ به ۴ بخش گسترش یافت", "unit": "R3", "status": "duplicate"}, {"text": "از ۸ شهر به ۲۹ شهر رسید", "unit": null, "status": "unsupported"}]
strict reason: R1-R3 correct; extra city list at launch; unsupported 8-city claim.
lax reason: All required claims are present and accurate; extra city list does not affect required facts.

## 24. HYBRID_15500 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: بودجهٔ سالانهٔ دولت سالانه شش کرور تومان کسری داشت (درآمد پانزده کرور و نیم و هزینه بیست‌ویک کرور و نیم تومان) و این کسری هر سال باید با وام پر می‌شد. کمیسیون مالی مجلس دو راه را در نظر گرفت: یکی افزودن بر درآمد دولت و دیگری کاستن از هزینه‌ها (دریافت‌ها). بر پایهٔ این دو راه، کمیسیون از هزینه‌های سالانه هشت کرور کم کرد؛ از جمله کاهش مستمری شاهزادگان و درباریان، برانداختن تیول و تسعیر، و الزام درباریان به پرداخت خودِ جنس مالیات. (گفتار چهارم و گفتار هشتم تاریخ مشروطه ایران، بندهای [2]، [3]، [11] و [16])
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد پانزده کرور و نیم و هزینه بیست‌ویک کرور و نیم تومان", "unit": null, "status": "extra"}, {"text": "این کسری هر سال باید با وام پر می‌شد", "unit": null, "status": "unsupported"}, {"text": "کمیسیون از هزینه‌های سالانه هشت کرور کم کرد", "unit": null, "status": "unsupported"}, {"text": "از جمله کاهش مستمری شاهزادگان و درباریان، برانداختن تیول و تسعیر، و الزام درباریان به پرداخت خودِ جنس مالیات", "unit": null, "status": "unsupported"}, {"text": "(گفتار چهارم و گفتار هشتم تاریخ مشروطه ایران، بندهای [2]، [3]، [11] و [16])", "unit": null, "status": "non_substantive"}]
strict reason: Core claims R1 and R2 correct; extra/unsupported details do not contradict.
lax reason: Core claims match required facts; additional details are harmless extras.

## 25. HYBRID_15500 V2-test-BOOK-fa-T1-005 (T1 fa) stricter=J1 structural=['claim_unsupported']
Q: When did Seyyed Zia's government fall and where did he go afterwards?
REF: On 4 Jowza 1300, after three months and some days; he went via Basra to Europe and stayed in Montreux, Switzerland.
ANS: دولت سید ضیاءالدین در 4 جوزا 1300 (بعد از سه ماه و چند روز) ساقط شد و او از راه بصره به اروپا رفت.

- طبق بخش «شاه از سید رنجیده و ازو میترسید»، پس از سه ماه و چند روز در 4 جوزا 1300 دولت او ساقط شد.
- در همان بخش آمده که تا سرحد مشایعت شد و از آنجا از راه بصره باروپا (اروپا) رفت.
- در مسیر، در «منترو» از شهرهای سویس متوقف گشت.
- بخش «شمه‌ای از کارهای سید ضیاءالدین» نیز تأیید می‌کند که او روز چهارشنبه 17 رمضان مطابق 4 جوزا 1300 با چند تن از همراهان بسوی قزوین حرکت کرد تا از راه بغداد به اروپا برود.
REQ: {"R1": "The government fell on 4 Jowza 1300.", "R2": "It lasted three months and some days.", "R3": "He went via Basra to Europe and stayed in Montreux, Switzerland."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "بخش «شمه‌ای از کارهای سید ضیاءالدین» نیز تأیید می‌کند که او روز چهارشنبه 17 رمضان مطابق 4 جوزا 1300 با چند تن از همراهان بسوی قزوین حرکت کرد تا از راه بغداد به اروپا برود", "unit": null, "status": "unsupported"}]
strict reason: All required facts are present; extra claim is unsupported but harmless.
lax reason: All required claims present; extra detail included but harmless.

## 26. HYBRID_15500 V2-test-BOOK-en-T3-005 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: What does Mill say majority feeling in the United States does to people with large incomes who would like to live showily, and how does Veblen's theory explain the motive that this feeling restrains?
REF: Mill: in the most democratic country, the majority's dislike of any more showy or costly style of living than they can rival works as a tolerably effectual sumptuary law, so that a person with a very large income finds it hard to spend it in any way that will not incur popular disapproval (he thinks the reports exaggerated but the result probable). Veblen: the end of accumulation is to rank high in pecuniary strength compared with the rest of the community, and conspicuous consumption of valuable goods is a means of reputability for the gentleman of leisure; showy spending is how wealth earns esteem.
ANS: On Liberty - Chapter IV: The feeling of the majority, to whom any appearance of a more showy or costly style of living than they can hope to rival is disagreeable, operates as a tolerably effectual sumptuary law, and in many parts of the Union it is really difficult for a person possessing a very large income to find any mode of spending it which will not incur popular disapprobation.
The Theory of the Leisure Class - Chapter Seven: Dress is an expression of pecuniary culture, and the wealthy leisure class sets the pace in matters of decency; conspicuous, costly display serves as a scheme of symbolism through which individuals seek honorific esteem, with subtler refinement arising as the leisure class grows.

Mill describes majority feeling in the United States as acting like a sumptuary law that makes it difficult for people with very large incomes to spend money showily without incurring popular disapprobation; Veblen's theory explains the restrained motive as the pursuit of honorific esteem through conspicuous, costly display, which the majority resents because those below cannot hope to rival it.
REQ: {"R1": "Mill: majority dislike of showy living works as a sumptuary law, making it hard for the rich to spend without popular disapproval.", "R2": "Veblen: conspicuous consumption of valuable goods is a means of reputability.", "R3": "Veblen: the aim is to rank high in pecuniary strength compared with others."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "The majority resents showy spending because those below cannot hope to rival it.", "unit": null, "status": "extra"}, {"text": "Dress is an expression of pecuniary culture, and the wealthy leisure class sets the pace in matters of decency.", "unit": null, "status": "unsupported"}, {"text": "Conspicuous display serves as a scheme of symbolism through which individuals seek honorific esteem.", "unit": "R2", "status": "duplicate"}]
strict reason: R1 and R2 correct; R3 missing; extra and unsupported claims present.
lax reason: Both required claims are accurately expressed; optional claim not provided.

## 27. HYBRID_15500 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['rule:cache']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: طبق اصل شصت و چهارم، اقلیت‌های دینی در مجموع پنج نماینده دارند:

1. زرتشتیان: یک نماینده
2. کلیمیان: یک نماینده
3. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
4. مسیحیان ارمنی جنوب: یک نماینده
5. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}]
strict reason: All required claims present except Jews; Klemian claim is extra; totals correct.
lax reason: All required claims match the reference exactly.

## 28. HYBRID_15500 V2-test-LAW-fa-T2-003 (T2 fa) stricter=J1 structural=['claim_unsupported']
Q: A property already entered in the property register was sold by an ordinary written agreement that was never registered with a notary. Under the Registration of Deeds and Property Act, was registration compulsory, and can the buyer rely on the agreement in court?
REF: Yes. Registration of deeds is optional in principle, but all contracts and transactions on the corpus or usufruct of property already in the property register are excepted (Art. 46(1)). A deed that had to be registered and was not is not accepted by any office or court (Art. 48), so under this Act the buyer cannot rely on the agreement in court.
ANS: **پاسخ:** بله، ثبت این معامله اجباری بود و خریدار نمی‌تواند در دادگاه به قولنامهٔ ثبت‌نشده استناد کند.

- طبق ماده 46 قانون ثبت اسناد و املاک، ثبت عقود و معاملات راجع به عین یا منافع املاکی که قبلاً در دفتر املاک ثبت شده، اجباری است (استثنا بر قاعدهٔ اختیاری بودن ثبت اسناد).
- طبق ماده 48، سندی که مطابق مواد فوق باید به ثبت برسد و به ثبت نرسیده، در هیچ‌یک از ادارات و محاکم پذیرفته نخواهد شد؛ بنابراین قولنامهٔ عادی ثبت‌نشده در دادگاه قابل استناد نیست.
- همچنین طبق ماده 22، دولت فقط کسی را مالک می‌شناسد که ملک به نام او ثبت شده یا انتقال به او در دفتر املاک به ثبت رسیده باشد.
REQ: {"R1": "Registration is optional in principle, except listed cases.", "R2": "Transactions on property already in the property register must be registered.", "R3": "An unregistered deed that had to be registered is not accepted by any office or court.", "R4": "The buyer cannot rely on the ordinary agreement in court under this Act."}
strict units: {'R1': 'missing', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'} | lax units: {'R1': 'missing', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): [{"text": "طبق ماده 46 ... اجباری است", "unit": "R2", "status": "duplicate"}, {"text": "طبق ماده 22، دولت فقط ... ثبت شده باشد", "unit": null, "status": "unsupported"}]
strict reason: Claims align with required conclusions; one extra unsupported reference; R1 not needed.
lax reason: R1 missing; R2, R3, R4 correct; extra detail on Art. 22.

## 29. HYBRID_15500 V2-test-LAW-fa-T2-004 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: A barber's permanent licence has expired; the union sent a notice but the owner did not renew, and he is advertising on Instagram. Under the Guild System Law as amended in 1392, what is the shop's status and what fine does the advertising carry?
REF: After the one-month renewal notice, a unit whose licence is not renewed is treated as unlicensed (Art. 5 note 2). Advertising in any way by a guild member without a valid licence is prohibited, with a fine of 1 million to 250 million rials (Art. 17 note 3).
ANS: پاسخ مستقیم: با انقضای پروانهٔ دائم و عدم تمدید پس از اخطاریهٔ یک‌ماهه، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود و انتشار آگهی تبلیغاتی توسط فرد فاقد پروانهٔ کسب معتبر ممنوع است؛ متخلف به جریمهٔ نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال محکوم می‌شود.

مبانی:
- طبق تبصرهٔ ۲ مادهٔ ۵، مدت اعتبار پروانهٔ کسب دائم پنج‌سال است و اتحادیه مکلف است با انقضای مدت اعتبار، اخطاریهٔ یک‌ماهه برای تمدید صادر کند (بند [1]).
- طبق همان تبصرهٔ ۲ مادهٔ ۵، در صورت عدم تبدیل یا تمدید پروانه، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود (بند [1]).
- طبق تبصرهٔ ۳ مادهٔ ۱۷، انتشار هرگونه آگهی تبلیغاتی توسط فرد صنفی فاقد پروانهٔ کسب معتبر ممنوع است و متخلف به جریمهٔ نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال محکوم می‌شود (بند [2]).
- همچنین طبق مادهٔ ۸۷ اصلاحی، فعالیت افراد صنفی در فضای مجازی مستلزم اخذ پروانهٔ کسب از اتحادیهٔ مربوطه است (بند [12]).
REQ: {"R1": "A unit whose licence is not renewed after the one-month notice is treated as unlicensed.", "R2": "Advertising by a guild member without a valid licence is prohibited.", "R3": "The fine is 1,000,000 to 250,000,000 rials."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "مدت اعتبار پروانهٔ کسب دائم پنج‌سال است", "unit": null, "status": "extra"}, {"text": "اتحادیه مکلف است با انقضای مدت اعتبار، اخطاریهٔ یک‌ماهه برای تمدید صادر کند", "unit": null, "status": "duplicate"}, {"text": "در صورت عدم تبدیل یا تمدید پروانه، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود", "unit": null, "status": "duplicate"}, {"text": "انتشار هرگونه آگهی تبلیغاتی توسط فرد صنفی فاقد پروانهٔ کسب معتبر ممنوع است و متخلف به جریمهٔ نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال محکوم می‌شود", "unit": null, "status": "duplicate"}, {"text": "فعالیت افراد صنفی در فضای مجازی مستلزم اخذ پروانهٔ کسب از اتحادیهٔ مربوطه است", "unit": null, "status": "unsupported"}, {"text": "بند [1]", "unit": null, "status": "non_substantive"}, {"text": "بند [2]", "unit": null, "status": "non_substantive"}, {"text": "بند [12]", "unit": null, "status": "non_substantive"}]
strict reason: All required claims correct; extra and unsupported claims do not alter the answer.
lax reason: All required claims match; extra statements do not affect correctness.

## 30. HYBRID_15500 V2-test-LAW-en-T3-001 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: Both the Digital Services Act and the Digital Markets Act use a 45 million user figure. What exactly does each Act count, and what else does the DMA require alongside that figure?
REF: DSA: 45 million or more average monthly active recipients of the service in the Union is the threshold for very large online platforms and search engines (Art. 33(1)). DMA: 45 million monthly active end users established or located in the Union in the last financial year, together with at least 10,000 yearly active business users established in the Union, triggers the presumption that a core platform service is an important gateway (Art. 3(2)(b)).
ANS: Digital Services Act - threshold: average monthly active recipients of the service in the Union equal to or higher than 45 million (equivalent to 10 % of the Union population), calculated as an average over a period of six months.
Digital Markets Act - threshold: at least 45 million monthly active end users established or located in the Union in the last financial year, identified and calculated in accordance with the methodology and indicators set out in the Annex.

Alongside the 45 million monthly active end users, the DMA also requires at least 10 000 yearly active business users established in the Union (and that the thresholds were met in each of the last three financial years). The two Acts are not directly comparable in measurement: the DSA counts average monthly active recipients over a six-month period, while the DMA counts monthly active end users over the financial year.
REQ: {"R1": "DSA counts average monthly active recipients of the service in the Union (>= 45 million).", "R2": "DMA counts monthly active end users established or located in the Union in the last financial year.", "R3": "DMA also requires at least 10,000 yearly active business users established in the Union.", "R4": "The DSA figure designates very large online platforms/search engines; the DMA figure triggers the gatekeeper 'important gateway' presumption."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "DSA calculation: average over a period of six months", "unit": null, "status": "extra"}, {"text": "DMA thresholds must be met in each of the last three financial years", "unit": null, "status": "unsupported"}, {"text": "DSA counts average monthly active recipients over a six-month period, DMA counts monthly active end users over the financial year", "unit": null, "status": "extra"}]
strict reason: R4 missing; unsupported claim about DMA's three-year requirement.
lax reason: Core requirements are present; extra details add unsupported methodology but do not contradict required facts.

## 31. HR_15500 V2-test-BOOK-fa-T1-005 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: When did Seyyed Zia's government fall and where did he go afterwards?
REF: On 4 Jowza 1300, after three months and some days; he went via Basra to Europe and stayed in Montreux, Switzerland.
ANS: دولت سیدضیاءالدین در 4 جوزا 1300 سقوط کرد و او از راه بصره به اروپا رفت و در «منترو» از شهرهای سوئیس متوقف شد.

- سقوط دولت پس از سه ماه و چند روز در 4 جوزا 1300 رخ داد (بند 1).
- در همان روز، دستخط شاه دایر بر عزل سیدضیاءالدین به عموم ایالات و حکام مخابره شد (بند 4).
- او تا سرحد مشایعت شد و از آنجا از راه بصره به اروپا رفت (بند 1).
- روز چهارشنبه 17 رمضان مطابق 4 جوزا 1300، چهار ساعت بظهر، با اپیکیان ارمنی، ماژور مسعودخان و کلنل کاظم‌خان و عده‌ای مستحفظ قزاق در سه اتومبیل به سوی قزوین حرکت کرد تا از راه بغداد به اروپا برود (بند 3).
REQ: {"R1": "The government fell on 4 Jowza 1300.", "R2": "It lasted three months and some days.", "R3": "He went via Basra to Europe and stayed in Montreux, Switzerland."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "سقوط دولت پس از سه ماه و چند روز در ۴ جوزا ۱۳۰۰ رخ داد", "unit": "R2", "status": "extra"}, {"text": "در همان روز، دستخط شاه دایر بر عزل سیدضیاءالدین به عموم ایالات و حکام مخابره شد", "unit": null, "status": "unsupported"}, {"text": "او تا سرحد مشایعت شد", "unit": null, "status": "extra"}, {"text": "روز چهارشنبه ۱۷ رمضان مطابق ۴ جوزا ۱۳۰۰، چهار ساعت بظهر، با اپیکیان ارمنی، ماژور مسعودخان و کلنل کاظم‌خان و عده‌ای مستحفظ قزاق در سه اتومبیل به سوی قزوین حرکت کرد تا از راه بغداد به اروپا برود", "unit": null, "status": "unsupported"}]
strict reason: R1 and R3 correct; R2 missing; unsupported details added.
lax reason: All required claims are present and accurate; optional claim correct; other statements are extra or duplicate.

## 32. HR_15500 V2-test-BOOK-fa-T3-002 (T3 fa) stricter=J2 structural=['claim_unsupported', 'req_missing']
Q: The man elected first vice-president of the first Majlis: what was his role in the 1919 agreement, and per Britain's later claim, what sum had been shared between him and two other ministers?
REF: Vosough al-Dowleh. He concluded the 1919 agreement with Britain (subject to Majlis approval). Per Britain's claim, finance minister Sarem al-Dowleh received £130,000 at the signing, and this sum was shared between Vosough al-Dowleh, Sarem al-Dowleh and Nosrat al-Dowleh.
ANS: نایب‌رئیس یکم نخستین مجلس، وثوق‌الدوله بود که بعدها قرارداد ۱۹۱۹ را با دولت بریتانیا منعقد کرد.

- در گشایش نخستین مجلس، صنیع‌الدوله رئیس و وثوق‌الدوله نایب‌رئیس یکم انتخاب شد (تاریخ مشروطه ایران، گفتار سوم).
- وثوق‌الدوله در سال دوم حکومتش با تقاضاهای جدید مواجه شد که به‌صورت قرارداد ۱۹۱۹ به دولت ایران پیشنهاد گردید و در ذی‌القعده ۱۳۳۷ (دسامبر ۱۹۱۹) این قرارداد بسته شد (قرارداد ۱۹۱۹ و احمدشاه).
- طبق ادعای دولت بریتانیا پس از انحلال پلیس جنوب، مبلغ صد و سی هزار لیره که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ شده بود، بین وثوق‌الدوله و صارم‌الدوله و نصرت‌الدوله تقسیم شده بود (آغاز ضعف دولت قوام).
REQ: {"R1": "The first vice-president of the first Majlis was Vosough al-Dowleh.", "R2": "He concluded the 1919 agreement with Britain.", "R3": "£130,000, received by Sarem al-Dowleh, was shared between Vosough, Sarem and Nosrat al-Dowleh."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "مبلغ صد و سی هزار لیره که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ شده بود، بین وثوق‌الدوله و صارم‌الدوله و نصرت‌الدوله تقسیم شده بود", "unit": "R3", "status": "unsupported"}, {"text": "در گشایش نخستین مجلس، صنیع‌الدوله رئیس و وثوق‌الدوله نایب‌رئیس یکم انتخاب شد", "unit": "R1", "status": "duplicate"}, {"text": "وثوق‌الدوله در سال دوم حکومتش با تقاضاهای جدید مواجه شد که به‌صورت قرارداد ۱۹۱۹ به دولت ایران پیشنهاد گردید و در ذی‌القعده ۱۳۳۷ (دسامبر ۱۹۱۹) این قرارداد بسته شد", "unit": null, "status": "extra"}, {"text": "طبق ادعای دولت بریتانیا پس از انحلال پلیس جنوب، مبلغ صد و سی هزار لیره...", "unit": null, "status": "unsupported"}, {"text": "(تاریخ مشروطه ایران، گفتار سوم)", "unit": null, "status": "non_substantive"}, {"text": "(قرارداد ۱۹۱۹ و احمدشاه)", "unit": null, "status": "non_substantive"}, {"text": "(آغاز ضعف دولت قوام)", "unit": null, "status": "non_substantive"}]
strict reason: R3 claim present but unsupported; explicit R3 statement missing.
lax reason: All required claims are present and accurate; one duplicate citation noted.

## 33. HR_15500 V2-test-BOOK-fa-T3-004 (T3 fa) stricter=J2 structural=['claim_contradiction']
Q: Returning from Europe in 1338 AH, who hosted Ahmad Shah at Muhammara before Basra? Per Kasravi, in the last year of his rule, how much did that host take in total from the Muhammara and Abadan contracts?
REF: Sheikh Khaz'al Khan. In the last year of his rule the Muhammara contract reached 120,000 tomans and the Abadan contract 210,000: 330,000 tomans in total.
ANS: میزبان، سردار اقدس (شیخ خزعل) بود (البته در متن، مهمانِ محمره ولیعهد است، نه شاه). کنترات محمره ۱۲۰ هزار تومان و کنترات آبادان ۲۱۰ هزار تومان بود، روی هم ۳۳۰ هزار تومان.
REQ: {"R1": "The host was Sheikh Khaz'al Khan.", "R2": "Muhammara contract: 120,000 tomans.", "R3": "Abadan contract: 210,000 tomans.", "R4": "Total 330,000 tomans."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): [{"text": "مهمانِ محمره ولیعهد است، نه شاه", "unit": null, "status": "contradiction"}]
strict reason: Contradiction about guest identity; rest matches requirements.
lax reason: All required and derived claims are present and accurate; extra comment does not affect answer.

## 34. HR_15500 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: طبق اصل شصت و چهارم، اقلیت‌های دینی در مجموع پنج نماینده دارند:
1. زرتشتیان: یک نماینده
2. کلیمیان: یک نماینده
3. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
4. مسیحیان ارمنی جنوب: یک نماینده
5. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}, {"text": "مسیحیان ارمنی شمال: یک نماینده", "unit": "R4", "status": "duplicate"}]
strict reason: All required claims present except Jews; extra claim about کلیمیان; total correctly derived.
lax reason: All required claims match the reference exactly.

## 35. HR_15500 V2-test-LAW-fa-T3-003 (T3 fa) stricter=J2 structural=['claim_wrong_binding']
Q: Under the 1345 Direct Taxes Law, what was the top income-tax rate (on annual taxable income above 50 million rials)? What did the 1351 amendment raise it to, and from when did the new rates apply?
REF: 55% on income above 50 million rials in the Art. 134 table of 1345; the 1351 amendment (item 44) replaced the table and set 60% on the excess over 50 million rials, for income earned from the start of 1352 (legal persons: fiscal years ending in 1352 or later).
ANS: قانون مالیات‌های مستقیم 1345 - ماده 133: از 50000000 ریال درآمد مشمول مالیات به بالا 55%
قانون اصلاح قانون مالیات‌های مستقیم 1351 - ماده 133 اصلاحی: از 50000000 ریال درآمد به بالا مالیات سالانه 60% نسبت به مازاد 50000000 ریال
قانون اصلاح قانون مالیات‌های مستقیم 1351 - زمان اعمال: نرخهای فوق شامل درآمدهایی که تاریخ تحصیل آن اول سال 1352 به بعد می‌باشد و همچنین مالیات بر درآمد اشخاص حقوقی مربوط به سال مالی که در سال 1352 به بعد خاتمه می‌یابد خواهد بود

در قانون 1345 بالاترین نرخ مالیات بر درآمد 55% بود که اصلاحیه 1351 آن را به 60% افزایش داد و نرخ‌های جدید شامل درآمدهایی با تاریخ تحصیل از اول سال 1352 به بعد (و سال مالی اشخاص حقوقی که در 1352 به بعد خاتمه می‌یابد) اعمال می‌شد.
REQ: {"R1": "Under 1345 the top rate (above 50 million rials) was 55%.", "R2": "Under the 1351 amendment the rate on the excess over 50 million rials is 60%.", "R3": "The new rates apply to income earned from the start of 1352 (legal persons: fiscal years ending 1352 or later).", "R4": "The top rate rose by 5 percentage points."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "ماده ۱۳۳ در قانون ۱۳۴۵ و اصلاحیه ۱۳۵۱ ذکر شده است.", "unit": null, "status": "wrong_binding"}]
strict reason: Mismatched article number (133 vs 134); derived comparison (R4) missing.
lax reason: All required factual claims present and accurate; optional derived claim not stated.

## 36. HR_15500 V2-test-LAW-en-T3-001 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: Both the Digital Services Act and the Digital Markets Act use a 45 million user figure. What exactly does each Act count, and what else does the DMA require alongside that figure?
REF: DSA: 45 million or more average monthly active recipients of the service in the Union is the threshold for very large online platforms and search engines (Art. 33(1)). DMA: 45 million monthly active end users established or located in the Union in the last financial year, together with at least 10,000 yearly active business users established in the Union, triggers the presumption that a core platform service is an important gateway (Art. 3(2)(b)).
ANS: Digital Services Act - threshold: 45 million average monthly active recipients of the service in the Union, equivalent to 10 % of the Union population.
Digital Markets Act - threshold: 45 million monthly active end users established or located in the Union in the last financial year.

The DMA additionally requires at least 10 000 yearly active business users established in the Union, and that the thresholds in point (b) were met in each of the last three financial years. The two figures are not directly comparable in measurement terms: the DSA counts average monthly active recipients of a service, while the DMA counts monthly active end users (unique users) plus yearly active business users under a methodology set out in its Annex.
REQ: {"R1": "DSA counts average monthly active recipients of the service in the Union (>= 45 million).", "R2": "DMA counts monthly active end users established or located in the Union in the last financial year.", "R3": "DMA also requires at least 10,000 yearly active business users established in the Union.", "R4": "The DSA figure designates very large online platforms/search engines; the DMA figure triggers the gatekeeper 'important gateway' presumption."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "equivalent to 10 % of the Union population", "unit": null, "status": "extra"}, {"text": "that the thresholds in point (b) were met in each of the last three financial years", "unit": null, "status": "unsupported"}, {"text": "The two figures are not directly comparable in measurement terms: the DSA counts average monthly active recipients of a service, while the DMA counts monthly active end users (unique users) plus yearl", "unit": null, "status": "extra"}]
strict reason: R4 omitted; unsupported 3-year requirement; extra details included.
lax reason: Core requirements are correct; extra unsupported details added.

## 37. HR_15500 V2-test-LAW-en-T3-002 (T3 en) stricter=J1 structural=['req_missing']
Q: Rank these maximum fines, expressed as a share of total worldwide annual turnover: the AI Act's fine for engaging in a prohibited AI practice, the Commission's fine on a very large online platform under the Digital Services Act, and the Commission's fine on a gatekeeper for breaching its Digital Markets Act obligations (first offence).
REF: DMA highest, then AI Act, then DSA. DMA: up to 10 % of total worldwide turnover in the preceding financial year (Art. 30(1)); AI Act: up to 7 % of total worldwide annual turnover or EUR 35 million, whichever is higher (Art. 99(3)); DSA: up to 6 % of total worldwide annual turnover (Art. 74(1)).
ANS: AI Act prohibited AI practice: 7 %; Digital Services Act (Commission, very large online platform): 6 %; Digital Markets Act (gatekeeper, first offence): 10 %
REQ: {"R1": "DMA: up to 10% of worldwide turnover.", "R2": "AI Act prohibited practices: up to 7% (or EUR 35 million, whichever is higher).", "R3": "DSA VLOP fine: up to 6% of worldwide annual turnover.", "R4": "Order from highest: DMA > AI Act > DSA."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): []
strict reason: All three numeric claims match required values; ranking requirement not provided.
lax reason: All values and implicit ranking match reference.

## 38. LEX5B_15500 V2-test-RPT-fa-T3-001 (T3 fa) stricter=J2 structural=['claim_wrong_value']
Q: How much did the total distance travelled by the Digi Express fleet change in 1403 compared with 1402?
REF: From 50 million km in 1402 to 54.6 million km in 1403: +4.6 million km (about 9%).
ANS: دیجی‌اکسپرس - 1402: 50 میلیون کیلومتر
دیجی‌اکسپرس - 1403: 54.6 میلیون کیلومتر

مجموع مسافت در سال 1403 نسبت به 1402 حدود 4.6 میلیون کیلومتر (حدود 9.2 درصد) افزایش یافته است.
REQ: {"R1": "Digi Express vehicles travelled 50 million km in 1402.", "R2": "Digi Express fleet travelled 54.6 million km in 1403.", "R3": "The increase was 4.6 million km (about 9%)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "حدود 9.2 درصد افزایش یافته است", "unit": "R3", "status": "wrong_value"}]
strict reason: R3's percentage (9.2%) does not match the required 9%.
lax reason: All required claims are present with accurate values; extra percentage detail does not affect correctness.

## 39. LEX5B_15500 V2-test-RPT-en-T2-005 (T2 en) stricter=J1 structural=['claim_unsupported']
Q: By how much did Lerøy's wild-catch volume fall in 2024, and what does the company say about the outlook for that business?
REF: Catch volume fell from 75,893 tonnes (HOG) in 2023 to 64,991 tonnes in 2024 - about 10,900 tonnes or 14% less. Lerøy says WildCatch earnings have been significantly affected by the low quota basis and expects 2025 and probably 2026 to be challenging.
ANS: Lerøy Havfisk's catch volume fell from 75 893 tonnes in 2023 to 64 991 tonnes in 2024 (a decrease of 10 902 tonnes, or roughly 14%).

- The fall reflects significant quota cuts: cod quotas were reduced by 34% and haddock by 43% year-on-year, plus a 6% reduction in the saithe quota in the northern zone (Results – Group, [1][2]).
- The quota reduction is only partly offset by higher prices, which help the trawler fleet but challenge the land-based industry; onshore whitefish processing made another loss in 2024, larger than in 2023 (Results – Group, [2]).
- On the outlook, the company says earnings in the WildCatch segment have been significantly affected by the low quota basis and will remain so for some time, with 2025 and probably also 2026 expected to be challenging for the segment (Results – Group / Market and outlook, [3][6]).
- Strategically, quotas for the highest-value species, cod, will fall further in 2025, so Lerøy aims to maintain efficient WildCatch operations while increasing onshore capacity utilisation by scaling up new processed products (Strategic priorities – WildCatch, [4]).
REQ: {"R1": "Catch volume was 64,991 tonnes (HOG) in 2024 versus 75,893 in 2023.", "R2": "The fall was about 10,900 tonnes (about 14%).", "R3": "Earnings are hit by the low quota basis and 2025 (probably 2026) are expected to be challenging."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "The fall reflects quota cuts: cod -34%, haddock -43%, saithe -6% in northern zone.", "unit": null, "status": "unsupported"}, {"text": "Quota reduction partly offset by higher prices; onshore whitefish processing loss larger in 2024.", "unit": null, "status": "unsupported"}, {"text": "Strategic plan: cod quotas will fall further in 2025; aim to maintain efficient WildCatch and boost onshore capacity.", "unit": null, "status": "extra"}]
strict reason: All required claims present and accurate; extra unsupported details do not affect required answer.
lax reason: All required claims correct; extra details do not alter core answer.

## 40. LEX5B_15500 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: بودجهٔ سالانهٔ دولت سالانه شش کرور تومان (و نیم) کسری داشت و کمیسیون مجلس دو راه‌حل را در نظر گرفت: یکی آنکه به درآمد دولت بیفزایند و دیگری آنکه از دریافت‌ها (مخارج) بکاهند.

- درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کسری می‌شد و می‌بایست با وام پر شود (گفتار چهارم، بندهای ۱ و ۷).
- کمیسیون برگزیدهٔ مجلس برای این کار (که وثوق‌الدوله یکی از اعضای آن بود) دو راه را اندیشید: افزودن به درآمد و کاستن از دریافت (بند ۲).
- برای افزونی درآمد، از جمله مقرر شد «تفاوت عمل» (فزونی مالیات که حکمرانان برداشت می‌کردند) به حساب دولت درآید (بند ۳).
- برای کاستن دررفت‌ها، کمیسیون از چهار راه سالانه هشت کرور کم کرد، از جمله کاستن از مستمری شاهزادگان (بند ۴).
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کسری می‌شد", "unit": null, "status": "extra"}, {"text": "می‌بایست با وام پر شود", "unit": null, "status": "unsupported"}, {"text": "کمیسیون برگزیدهٔ مجلس برای این کار (که وثوق‌الدوله یکی از اعضای آن بود) دو راه را اندیشید", "unit": null, "status": "extra"}, {"text": "برای افزونی درآمد، از جمله مقرر شد «تفاوت عمل» به حساب دولت درآید", "unit": null, "status": "unsupported"}, {"text": "برای کاستن دررفت‌ها، کمیسیون از چهار راه سالانه هشت کرور کم کرد، از جمله کاستن از مستمری شاهزادگان", "unit": null, "status": "unsupported"}]
strict reason: Core claims match requirements; extra details unsupported but do not contradict.
lax reason: All required claims are present and accurate; additional statements are non‑essential extras or duplicates.

## 41. LEX5B_15500 V2-test-BOOK-fa-T3-002 (T3 fa) stricter=J2 structural=['claim_unsupported']
Q: The man elected first vice-president of the first Majlis: what was his role in the 1919 agreement, and per Britain's later claim, what sum had been shared between him and two other ministers?
REF: Vosough al-Dowleh. He concluded the 1919 agreement with Britain (subject to Majlis approval). Per Britain's claim, finance minister Sarem al-Dowleh received £130,000 at the signing, and this sum was shared between Vosough al-Dowleh, Sarem al-Dowleh and Nosrat al-Dowleh.
ANS: نایب‌رئیس یکم مجلس اول، وثوق‌الدوله بود که بعدها نخست‌وزیر شد و قرارداد ۱۹۱۹ را با دولت بریتانیا منعقد کرد و مالیه و قشون ایران را زیر نظر مستشاران انگلیسی قرار داد. طبق مطالبهٔ دولت بریتانیا در زمان قوام‌السلطنه، مبلغ صد و سی هزار لیره که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ شده بود، بین وثوق‌الدوله و صارم‌الدوله و نصرةالدوله تقسیم شده بود.

- وثوق‌الدوله نایب رئیس یکم مجلس اول بود (گفتار سوم، تاریخ مشروطه ایران).
- دولت وثوق‌الدوله در ۱۳ اسد ۱۳۳۶ (اوت ۱۹۱۸) تشکیل شد و قرارداد معروف در ذی‌القعده ۱۳۳۷ (دسامبر ۱۹۱۹) با بریتانیا بسته شد که مالیه و قشون ایران زیر نظر معلمین و فرماندهان انگلیسی درآمد و «آرمیتاژ اسمیت» و «ژنرال دیکسن» وارد ایران شدند (بخش قرارداد ۱۹۱۹ و احمدشاه).
- در قضیهٔ انحلال پلیس جنوب، دولت بریتانیا وجوه مصرف‌شده در امنیت جنوب را مطالبه کرد، از جمله صد و سی هزار لیره‌ای که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ و بین وثوق‌الدوله و صارم‌الدوله و نصرةالدوله تقسیم شده بود (بخش آغاز ضعف دولت قوام).
REQ: {"R1": "The first vice-president of the first Majlis was Vosough al-Dowleh.", "R2": "He concluded the 1919 agreement with Britain.", "R3": "£130,000, received by Sarem al-Dowleh, was shared between Vosough, Sarem and Nosrat al-Dowleh."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "قرارداد ۱۹۱۹ مالیه و قشون ایران را زیر نظر مستشاران انگلیسی قرار داد", "unit": null, "status": "extra"}, {"text": "دولت وثوق‌الدوله در ۱۳ اسد ۱۳۳۶ (اوت ۱۹۱۸) تشکیل شد", "unit": null, "status": "extra"}, {"text": "قرارداد معروف در ذی‌القعده ۱۳۳۷ (دسامبر ۱۹۱۹) با بریتانیا بسته شد", "unit": null, "status": "extra"}, {"text": "در قضیهٔ انحلال پلیس جنوب، دولت بریتانیا وجوه مصرف‌شده در امنیت جنوب را مطالبه کرد", "unit": null, "status": "unsupported"}]
strict reason: All required claims correct; extra details do not contradict.
lax reason: All required claims are present and accurate; no contradictions or missing information.

## 42. LEX5B_15500 V2-test-BOOK-en-T2-001 (T2 en) stricter=J1 structural=['claim_wrong_value', 'req_wrong']
Q: Under Mill's principle of liberty, may someone forcibly stop a person who is about to cross a bridge known to be unsafe? Explain when force is allowed, when only a warning is, and why this does not breach his liberty; and say to whom Mill says the principle does not apply at all.
REF: If there is no time to warn him, anyone may seize him and turn him back without any real infringement of his liberty, because liberty is doing what one desires and he does not desire to fall into the river. When there is not certainty but only a danger of mischief, he should only be warned, not forcibly stopped (unless he is a child, delirious, or unable to reflect). The principle does not apply at all to children and young persons below the legal age of manhood or womanhood, to others who must be taken care of, or to backward states of society (barbarians), for whom despotism aimed at their improvement is legitimate.
ANS: Yes, force is allowed in the bridge case. According to Chapter V (Applications), if a public officer or anyone else sees a person attempting to cross a bridge ascertained to be unsafe, and there is no time to warn him, they may seize him and turn him back. This does not really infringe his liberty because liberty consists in doing what one desires, and he does not desire to fall into the river. However, when there is only a danger of mischief rather than certainty, no one but the person himself can judge the sufficiency of the motive for incurring the risk, so (unless he is a child, delirious, or in some state of excitement or absorption incompatible with the full use of the reflecting faculty) he ought only to be warned of the danger, not forcibly prevented (Chapter V). Finally, Mill states in Chapter I (Introductory) that the doctrine applies only to human beings in the maturity of their faculties: it does not apply to children or young persons below the legal age, who must be protected against their own actions as well as against external injury.
REQ: {"R1": "The rule: power may be used against a person's will only to prevent harm to others.", "R2": "If there is no time to warn, he may be seized and turned back.", "R3": "This is no infringement because liberty is doing what one desires and he does not desire to fall.", "R4": "With only a danger (not certainty), he should only be warned (unless a child, delirious, etc.).", "R5": "The principle does not apply to children/minors, those who need to be cared for, or backward states of society (barbarians)."}
strict units: {'R1': 'missing', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'wrong'} | lax units: {'R1': 'missing', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "The principle does not apply to children or young persons below the legal age.", "unit": "R5", "status": "wrong_value"}]
strict reason: All required claims present except R5 incomplete; R1 optional omitted.
lax reason: Missing the general rule (R1); all other required claims correct.

## 43. LEX5B_15500 V2-test-BOOK-en-T3-003 (T3 en) stricter=J1 structural=['claim_unsupported', 'req_wrong']
Q: Mill and Tocqueville both warn against the tyranny of the majority. Beyond laws and public officials, what does each see as its most dangerous reach?
REF: Mill: society itself becomes the tyrant through prevailing opinion and feeling, a social tyranny more formidable than many kinds of political oppression because it leaves fewer means of escape, penetrates into the details of life and enslaves the soul; so protection is needed against the tyranny of prevailing opinion, not only of the magistrate. Tocqueville: the American majority's power is both physical and moral; it acts on the will as well as on actions and represses not only contest but controversy, so that he knows no country with so little true independence of mind and freedom of discussion as America.
ANS: Mill - On Liberty: Society itself can tyrannise beyond political functionaries, imposing "its own ideas and practices as rules of conduct on those who dissent from them" through prevailing opinion and feeling, penetrating "much more deeply into the details of life, and enslaving the soul itself" [1][7].

Tocqueville - Democracy in America: The majority's power reaches through public opinion, the legislature, the executive, the troops, the jury, and even elected judges, so that the wronged individual finds "no sure barrier" and no avenue of redress; its effects are felt "more sensibly... in the manners than in the conduct of society," checking the development of leading characters [3][6][8].

Comparison: Mill sees the deepest danger as social tyranny—the majority's moral and opinion-based pressure on individual thought and self-regarding conduct, which leaves fewer means of escape than political oppression [1][10]. Tocqueville sees the danger as the majority's overpowering strength across all institutions and manners, leaving inadequate securities against tyranny even where laws mitigate it [6][8]. Both agree the threat extends beyond laws and officials into opinion itself, but Mill stresses enslavement of the soul through social conformity, while Tocqueville stresses the majority's unchecked institutional and moral dominance in America.
REQ: {"R1": "Mill: social tyranny of prevailing opinion, reaching the details of life and enslaving the soul.", "R2": "Mill: protection is needed against prevailing opinion, not only the magistrate.", "R3": "Tocqueville: the majority acts on the will and thought, repressing controversy; little independence of mind in America."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct'}
strict claims (non-correct): [{"text": "Tocqueville: majority power reaches through public opinion, legislature, executive, troops, jury, judges, leaving no barrier for the wronged individual.", "unit": "R3", "status": "unsupported"}, {"text": "Comparison summarising Mill’s social tyranny vs. Tocqueville’s institutional/moral dominance.", "unit": null, "status": "extra"}]
strict reason: Mill claim matches required R1; Tocqueville claim does not reflect R3; optional R2 not addressed.
lax reason: R1 and R3 correct; R2 missing; extra details and citations ignored.

## 44. LEX5B_15500 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: طبق اصل شصت و چهارم، اقلیت‌های دینی در مجموع پنج نماینده دارند:

1. زرتشتیان: یک نماینده
2. کلیمیان: یک نماینده
3. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
4. مسیحیان ارمنی جنوب: یک نماینده
5. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}]
strict reason: All required claims present except Jews; Klemian claim is extra; totals correct.
lax reason: All required claims present and accurate.

## 45. LEX5B_15500 V2-test-LAW-en-T2-001 (T2 en) stricter=J1 structural=['claim_unsupported']
Q: A website's sign-up form has a pre-ticked box saying the user agrees to marketing profiling, and its terms say consent can only be withdrawn by posting a signed letter. Under the GDPR, is the pre-ticked box valid consent, and is the letter-only withdrawal acceptable?
REF: Neither is acceptable. Consent must be an unambiguous indication given by a statement or a clear affirmative action (Art. 4(11)), and silence, pre-ticked boxes or inactivity do not constitute consent (recital 32). The data subject may withdraw consent at any time, and it must be as easy to withdraw as to give consent (Art. 7(3)), so a letter-only withdrawal for consent given online does not meet that rule.
ANS: No — both fail under the GDPR.

- Recital 32 of Regulation (EU) 2016/679 states that consent must be a clear affirmative act, and that silence, pre-ticked boxes or inactivity should not constitute consent — so the pre-ticked box is invalid [2].
- Article 7(3) gives the data subject the right to withdraw consent at any time, and it must be as easy to withdraw as to give consent — a signed-letter-only rule does not meet that standard [4].
- Withdrawal does not affect the lawfulness of processing before withdrawal, and the data subject must be informed of this before giving consent [3].
- Article 7(2) also requires that consent requests in broader written declarations be clearly distinguishable, intelligible and in clear and plain language, with infringing parts not binding [4].
REQ: {"R1": "Consent requires a statement or clear affirmative action; pre-ticked boxes are not consent.", "R2": "Withdrawal must be possible at any time and as easy as giving consent.", "R3": "A letter-only withdrawal for consent given by an online tick does not satisfy Art. 7(3).", "R4": "Consent needs a statement or clear affirmative action; pre-ticked boxes are not consent."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "Article 7(2) requires consent requests to be clear and plain‑language and that infringing parts are not binding", "unit": null, "status": "unsupported"}]
strict reason: All required conclusions matched; optional rule restated; extra claim not in packet.
lax reason: Core claims match requirements; extra details are harmless.

## 46. LEX5B_15500 V2-test-LAW-en-T3-001 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: Both the Digital Services Act and the Digital Markets Act use a 45 million user figure. What exactly does each Act count, and what else does the DMA require alongside that figure?
REF: DSA: 45 million or more average monthly active recipients of the service in the Union is the threshold for very large online platforms and search engines (Art. 33(1)). DMA: 45 million monthly active end users established or located in the Union in the last financial year, together with at least 10,000 yearly active business users established in the Union, triggers the presumption that a core platform service is an important gateway (Art. 3(2)(b)).
ANS: Digital Services Act - 45 million threshold: average monthly active recipients of the service in the Union equal to or higher than 45 million, equivalent to 10 % of the Union population, calculated as an average over a period of six months, for designation as very large online platforms or very large online search engines (Art. 33(1); recital).

Digital Markets Act - 45 million threshold: at least 45 million monthly active end users established or located in the Union in the last financial year, identified and calculated in accordance with the methodology and indicators set out in the Annex (Art. 3(1)(b)).

Digital Markets Act - additional requirement alongside the 45 million: at least 10 000 yearly active business users established in the Union, and the thresholds in point (b) must have been met in each of the last three financial years (Art. 3(1)(b) and (c)).

Both Acts use a 45 million monthly user figure, but the DSA counts average monthly active recipients of an online platform or search engine service (a six-month average, ~10 % of the Union population), while the DMA counts monthly active end users of a core platform service; the DMA additionally requires at least 10 000 yearly active business users and that the thresholds were met in each of the last three financial years.
REQ: {"R1": "DSA counts average monthly active recipients of the service in the Union (>= 45 million).", "R2": "DMA counts monthly active end users established or located in the Union in the last financial year.", "R3": "DMA also requires at least 10,000 yearly active business users established in the Union.", "R4": "The DSA figure designates very large online platforms/search engines; the DMA figure triggers the gatekeeper 'important gateway' presumption."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "DSA threshold equivalent to 10 % of the Union population, calculated as an average over a period of six months", "unit": null, "status": "extra"}, {"text": "DMA requires thresholds in point (b) to have been met in each of the last three financial years", "unit": null, "status": "unsupported"}, {"text": "DSA counts average monthly active recipients (six-month average, ~10 % of the Union population)", "unit": "R1", "status": "duplicate"}, {"text": "DMA counts monthly active end users of a core platform service", "unit": "R2", "status": "duplicate"}, {"text": "DMA additionally requires at least 10,000 yearly active business users", "unit": "R3", "status": "duplicate"}]
strict reason: DMA three-year requirement unsupported; R4 legal consequence missing.
lax reason: All required facts are correctly stated; extra details are harmless; no contradictions.

## 47. LEX5B_15500 V2-test-LAW-en-T3-002 (T3 en) stricter=J2 structural=['claim_wrong_value', 'req_wrong']
Q: Rank these maximum fines, expressed as a share of total worldwide annual turnover: the AI Act's fine for engaging in a prohibited AI practice, the Commission's fine on a very large online platform under the Digital Services Act, and the Commission's fine on a gatekeeper for breaching its Digital Markets Act obligations (first offence).
REF: DMA highest, then AI Act, then DSA. DMA: up to 10 % of total worldwide turnover in the preceding financial year (Art. 30(1)); AI Act: up to 7 % of total worldwide annual turnover or EUR 35 million, whichever is higher (Art. 99(3)); DSA: up to 6 % of total worldwide annual turnover (Art. 74(1)).
ANS: Artificial Intelligence Act (prohibited AI practice, Art. 99(3)): 7 % of total worldwide annual turnover
Digital Services Act (Commission fine on very large online platform, Art. 74(1)): 6 % of total worldwide annual turnover
Digital Markets Act (Commission fine on gatekeeper, first offence, Art. 30(1)): 10 % of total worldwide annual turnover

Ranked from lowest to highest, the AI Act's 7 % exceeds the DSA's 6 %, while the DMA's 10 % for a first offence is the highest of the three.
REQ: {"R1": "DMA: up to 10% of worldwide turnover.", "R2": "AI Act prohibited practices: up to 7% (or EUR 35 million, whichever is higher).", "R3": "DSA VLOP fine: up to 6% of worldwide annual turnover.", "R4": "Order from highest: DMA > AI Act > DSA."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct'}
strict claims (non-correct): [{"text": "Ranked from lowest to highest, the AI Act's 7 % exceeds the DSA's 6 %, while the DMA's 10 % for a first offence is the highest of the three", "unit": "R4", "status": "wrong_value"}]
strict reason: Ranking order reversed; should be DMA > AI Act > DSA.
lax reason: All claims match required values and ranking.

## 48. SETR_6000 V2-test-RPT-fa-T2-002 (T2 fa) stricter=J1 structural=['claim_wrong_value']
Q: In 1403, what percentage of Digikala's sellers were in cities other than Tehran?
REF: About 95%: 444,203 of 466,169 sellers.
ANS: حدود ۹۵٫۳ درصد (۴۴۴,۲۰۳ فروشنده از مجموع ۴۶۶,۱۶۹ فروشنده در سال ۱۴۰۳ در شهرهایی غیر از تهران بودند).
REQ: {"R1": "Digikala had 466,169 sellers in 1403.", "R2": "444,203 sellers were in cities other than Tehran.", "R3": "About 95% of sellers were outside Tehran."}
strict units: {'R1': 'correct', 'R2': 'wrong', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "۴۴۴,۲۰۲ فروشنده در شهرهایی غیر از تهران", "unit": "R2", "status": "wrong_value"}]
strict reason: Percentage matches R3; R2 off by one seller; R1 correctly stated.
lax reason: All required claims correct; operands included as extra detail.

## 49. SETR_6000 V2-test-RPT-fa-T3-001 (T3 fa) stricter=J2 structural=['claim_wrong_value']
Q: How much did the total distance travelled by the Digi Express fleet change in 1403 compared with 1402?
REF: From 50 million km in 1402 to 54.6 million km in 1403: +4.6 million km (about 9%).
ANS: دیجی‌اکسپرس - 1402: 50 میلیون کیلومتر
دیجی‌اکسپرس - 1403: 54.6 میلیون کیلومتر

مجموع مسافت در سال 1403 نسبت به 1402 حدود 4.6 میلیون کیلومتر افزایش یافته است (54.6 − 50 = 4.6 میلیون کیلومتر، یعنی حدود 9.2٪ رشد).
REQ: {"R1": "Digi Express vehicles travelled 50 million km in 1402.", "R2": "Digi Express fleet travelled 54.6 million km in 1403.", "R3": "The increase was 4.6 million km (about 9%)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "54.6 − 50 = 4.6 میلیون کیلومتر", "unit": "R3", "status": "extra"}, {"text": "حدود 9.2٪ رشد", "unit": "R3", "status": "wrong_value"}]
strict reason: R3's percentage (9.2%) contradicts reference (9%).
lax reason: All claims match required facts and derived increase accurately.

## 50. SETR_6000 V2-test-RPT-en-T2-002 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: What return on invested capital (as reported, not the underlying ROIC) did A.P. Moller - Maersk report for the group for 2024, how did it compare with its yearly target, and how does Maersk calculate this measure?
REF: 12.3% (last 12 months), above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
ANS: A.P. Moller - Maersk reported a group ROIC (LTM) of 12.3% for 2024, which was above the yearly target of above 7.5% under normalised conditions. Profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target. The measure is defined as profit/loss before financial items for the year (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
REQ: {"R1": "Group ROIC (LTM) was 12.3% in 2024.", "R2": "It was above the yearly target of above 7.5%.", "R3": "ROIC = EBIT less tax on EBIT divided by average invested capital, last twelve months."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "Profitability lagged in the first half of 2024, but the strong second-half result brought ROIC above target.", "unit": null, "status": "unsupported"}]
strict reason: All required claims correct; one unsupported detail.
lax reason: All candidate claims match required facts precisely; no missing, extra, or contradictory statements.

## 51. SETR_6000 V2-test-BOOK-en-T3-005 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: What does Mill say majority feeling in the United States does to people with large incomes who would like to live showily, and how does Veblen's theory explain the motive that this feeling restrains?
REF: Mill: in the most democratic country, the majority's dislike of any more showy or costly style of living than they can rival works as a tolerably effectual sumptuary law, so that a person with a very large income finds it hard to spend it in any way that will not incur popular disapproval (he thinks the reports exaggerated but the result probable). Veblen: the end of accumulation is to rank high in pecuniary strength compared with the rest of the community, and conspicuous consumption of valuable goods is a means of reputability for the gentleman of leisure; showy spending is how wealth earns esteem.
ANS: On Liberty (Mill) - United States: the feeling of the majority, to whom any appearance of a more showy or costly style of living than they can hope to rival is disagreeable, operates as a tolerably effectual sumptuary law, making it difficult for a person with a very large income to find any mode of spending it that will not incur popular disapprobation.
The Theory of the Leisure Class (Veblen) - dress/consumption: dress and consumption serve as symbolic expression of pecuniary culture, where display confers honorific esteem from one's own class, and the wealthy leisure class sets the pace in matters of decency.

Mill describes majority feeling in the United States as acting like a sumptuary law that restrains conspicuous, showy spending by the very wealthy, since such display incurs popular disapprobation. Veblen's theory explains the motive being restrained: the desire to display wealth through costly living and dress in order to win honorific esteem, a pecuniary form of status competition.
REQ: {"R1": "Mill: majority dislike of showy living works as a sumptuary law, making it hard for the rich to spend without popular disapproval.", "R2": "Veblen: conspicuous consumption of valuable goods is a means of reputability.", "R3": "Veblen: the aim is to rank high in pecuniary strength compared with others."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "Veblen: dress and consumption serve as symbolic expression of pecuniary culture, conferring honorific esteem.", "unit": "R2", "status": "extra"}, {"text": "Veblen: the wealthy leisure class sets the pace in matters of decency.", "unit": null, "status": "unsupported"}]
strict reason: R1 and R2 correct; R3 missing; minor unsupported detail.
lax reason: All required claims are accurately covered; optional claim omitted; one harmless extra statement added.

## 52. SETR_6000 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: ۱. مجموع نمایندگان اقلیت‌های دینی: پنج نفر
۲. زرتشتیان: یک نماینده
۳. کلیمیان: یک نماینده
۴. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
۵. مسیحیان ارمنی جنوب: یک نماینده
۶. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}]
strict reason: All required claims present except Jews; extra claim for Klemian; totals and individual seats match.
lax reason: All required claims correctly stated and matched.

## 53. SETR_6000 V2-test-LAW-en-T2-001 (T2 en) stricter=J1 structural=['claim_unsupported']
Q: A website's sign-up form has a pre-ticked box saying the user agrees to marketing profiling, and its terms say consent can only be withdrawn by posting a signed letter. Under the GDPR, is the pre-ticked box valid consent, and is the letter-only withdrawal acceptable?
REF: Neither is acceptable. Consent must be an unambiguous indication given by a statement or a clear affirmative action (Art. 4(11)), and silence, pre-ticked boxes or inactivity do not constitute consent (recital 32). The data subject may withdraw consent at any time, and it must be as easy to withdraw as to give consent (Art. 7(3)), so a letter-only withdrawal for consent given online does not meet that rule.
ANS: No — both fail under the GDPR.

- Recital 32 states consent must be a clear affirmative act, and silence, pre-ticked boxes or inactivity should not constitute consent, so the pre-ticked box is not valid consent.
- Article 7(3) gives the data subject the right to withdraw consent at any time, and it must be as easy to withdraw as to give consent, so requiring a posted signed letter is not acceptable.
- Recital 42 adds that consent is not freely given if the data subject is unable to refuse or withdraw consent without detriment, reinforcing that the letter-only withdrawal condition is invalid.
REQ: {"R1": "Consent requires a statement or clear affirmative action; pre-ticked boxes are not consent.", "R2": "Withdrawal must be possible at any time and as easy as giving consent.", "R3": "A letter-only withdrawal for consent given by an online tick does not satisfy Art. 7(3).", "R4": "Consent needs a statement or clear affirmative action; pre-ticked boxes are not consent."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "Recital 42 reinforces that the letter‑only condition is invalid", "unit": null, "status": "unsupported"}]
strict reason: Claims match required conclusions; extra Recital 42 not in packet; optional units not addressed.
lax reason: Core claims correct; extra detail harmless; missing operands not required.

## 54. SETR_6000 V2-test-LAW-en-T2-002 (T2 en) stricter=J1 structural=['claim_unsupported']
Q: A university lab builds an AI system that infers workers' emotions from webcam video, developed and put into service solely for scientific research. Later a retailer buys the same system to monitor staff mood in its shops. Under the AI Act, is either use caught by the prohibition on emotion recognition?
REF: The lab's system is outside the AI Act altogether: the Regulation does not apply to AI systems specifically developed and put into service for the sole purpose of scientific research and development (Art. 2(6)). The retailer's use is prohibited: the use of AI systems to infer emotions of a natural person in the workplace is a banned practice unless it is intended for medical or safety reasons, which staff-mood monitoring is not (Art. 5(1)(f)).
ANS: **Yes — the retailer's use is caught; the lab's research use is not.**

- Article 5(f) prohibits placing on the market, putting into service, or using AI systems that infer emotions of natural persons in the workplace and education institutions [1][7].
- The prohibition does not apply to AI systems intended for medical or safety reasons [1][2].
- The Regulation excludes from its scope AI systems specifically developed and put into service for the sole purpose of scientific research and development, so the lab's research use falls outside the prohibition [3].
- However, once such a system is placed on the market or put into service as a result of that activity, the obligation to comply with the Regulation applies [3].
- The retailer's use to monitor staff mood in shops is workplace emotion inference, not for medical or safety reasons, so it is prohibited under Article 5(f) [1][2].
REQ: {"R1": "The research-only system is outside the Regulation.", "R2": "Workplace emotion inference is prohibited except for medical or safety reasons.", "R3": "The retailer's staff-mood monitoring is a prohibited practice."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "the Regulation excludes AI systems developed for sole scientific research purpose", "unit": "R1", "status": "duplicate"}, {"text": "once such a system is placed on the market or put into service, the obligation to comply with the Regulation applies", "unit": null, "status": "unsupported"}, {"text": "citation markers like [1][2][3][7]", "unit": null, "status": "non_substantive"}]
strict reason: All required claims are correctly covered; optional claim matches; extra unsupported statement flagged.
lax reason: All required claims correctly stated; extra detail harmless.

## 55. SETR_6000 V2-test-LAW-en-T2-004 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: A platform company's core platform service crosses all the Digital Markets Act gatekeeper thresholds, but the company does not tell the Commission. How long did it have to notify, and what is the maximum fine for failing to do so?
REF: It had to notify the Commission without delay and in any event within 2 months after the thresholds were met (Art. 3(3)). Failing, intentionally or negligently, to comply with that notification obligation can be fined up to 1 % of its total worldwide turnover in the preceding financial year (Art. 30(3)(b)).
ANS: It had to notify within 2 months, and the maximum fine is up to 1 % of its total worldwide turnover in the preceding financial year.

- Article 3(3) requires an undertaking whose core platform service meets all the thresholds in paragraph 2 to notify the Commission without delay and in any event within 2 months after the thresholds are met.
- Article 3(4) (second subparagraph of [2]) says that if the undertaking fails to notify and fails to provide the information requested under Article 21, the Commission may still designate it as a gatekeeper based on information available to the Commission.
- Article 30(3) allows the Commission to impose fines not exceeding 1 % of total worldwide turnover in the preceding financial year on undertakings that intentionally or negligently breach obligations such as the notification duty.
- The 20 % fine in Article 30(2) applies only to repeated non-compliance with Articles 5, 6 or 7, not to failure to notify.
REQ: {"R1": "Notification is due within 2 months after the thresholds are met.", "R2": "The maximum fine for failing to notify is 1% of total worldwide turnover in the preceding financial year."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "Article 3(3) requires an undertaking whose core platform service meets all the thresholds in paragraph 2 to notify the Commission without delay and in any event within 2 months after the thresholds ar", "unit": "R1", "status": "extra"}, {"text": "Article 3(4) (second subparagraph of [2]) says that if the undertaking fails to notify and fails to provide the information requested under Article 21, the Commission may still designate it as a gatek", "unit": null, "status": "unsupported"}, {"text": "Article 30(3) allows the Commission to impose fines not exceeding 1 % of total worldwide turnover in the preceding financial year on undertakings that intentionally or negligently breach obligations s", "unit": "R2", "status": "extra"}, {"text": "The 20 % fine in Article 30(2) applies only to repeated non-compliance with Articles 5, 6 or 7, not to failure to notify", "unit": null, "status": "unsupported"}]
strict reason: Core claims correct; extra and unsupported details do not alter required facts.
lax reason: All required claims are present and accurate; extra statements do not affect correctness.

## 56. SETR_15500 V2-test-RPT-fa-T3-001 (T3 fa) stricter=J1 structural=['claim_wrong_value']
Q: How much did the total distance travelled by the Digi Express fleet change in 1403 compared with 1402?
REF: From 50 million km in 1402 to 54.6 million km in 1403: +4.6 million km (about 9%).
ANS: دیجی‌اکسپرس - 1402: 50 میلیون کیلومتر
دیجی‌اکسپرس - 1403: 54.6 میلیون کیلومتر

مجموع مسافت در سال 1403 نسبت به 1402 حدود 4.6 میلیون کیلومتر افزایش یافته است (رشد حدود 9.2 درصد).
REQ: {"R1": "Digi Express vehicles travelled 50 million km in 1402.", "R2": "Digi Express fleet travelled 54.6 million km in 1403.", "R3": "The increase was 4.6 million km (about 9%)."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'wrong'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "مجموع مسافت در سال 1403 نسبت به 1402 حدود 4.6 میلیون کیلومتر افزایش یافته است (رشد حدود 9.2 درصد)", "unit": "R3", "status": "wrong_value"}]
strict reason: km values match; percentage deviates from expected ~9%.
lax reason: All required claims correct; 9.2% is extra but harmless.

## 57. SETR_15500 V2-test-RPT-en-T2-002 (T2 en) stricter=J2 structural=['claim_unsupported']
Q: What return on invested capital (as reported, not the underlying ROIC) did A.P. Moller - Maersk report for the group for 2024, how did it compare with its yearly target, and how does Maersk calculate this measure?
REF: 12.3% (last 12 months), above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.
ANS: A.P. Moller - Maersk reported a group ROIC of 12.3% for 2024, which was above the yearly target of above 7.5% under normalised conditions. Maersk calculates ROIC as profit/loss before financial items for the year (EBIT) less tax on EBIT, divided by the average invested capital over the last twelve months.

- The 12.3% figure and the comparison with the >7.5% yearly target come from the Roadmap towards 2025 section [1], which also notes profitability lagged in the first half of 2024 but the strong second half brought ROIC above target.
- The same 12.3% (up from 7.4% in 2023) is confirmed in the Financial review [3] and the Five-year summary [6].
- The definition of the measure is given in the Definition of terms section [2]/[12].
REQ: {"R1": "Group ROIC (LTM) was 12.3% in 2024.", "R2": "It was above the yearly target of above 7.5%.", "R3": "ROIC = EBIT less tax on EBIT divided by average invested capital, last twelve months."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "The 12.3% figure and the comparison with the >7.5% yearly target come from the Roadmap towards 2025 section [1]", "unit": null, "status": "extra"}, {"text": "which also notes profitability lagged in the first half of 2024 but the strong second half brought ROIC above target", "unit": null, "status": "unsupported"}, {"text": "The same 12.3% (up from 7.4% in 2023) is confirmed in the Financial review [3] and the Five-year summary [6]", "unit": null, "status": "extra"}, {"text": "The definition of the measure is given in the Definition of terms section [2]/[12]", "unit": null, "status": "non_substantive"}]
strict reason: All required claims correct; extra/unsupported details do not alter core answer.
lax reason: All required claims are present and accurate; extra details do not affect correctness.

## 58. SETR_15500 V2-test-RPT-en-T2-005 (T2 en) stricter=J1 structural=['claim_unsupported']
Q: By how much did Lerøy's wild-catch volume fall in 2024, and what does the company say about the outlook for that business?
REF: Catch volume fell from 75,893 tonnes (HOG) in 2023 to 64,991 tonnes in 2024 - about 10,900 tonnes or 14% less. Lerøy says WildCatch earnings have been significantly affected by the low quota basis and expects 2025 and probably 2026 to be challenging.
ANS: Lerøy's catch volume fell from 75 893 tonnes (HOG) in 2023 to 64 991 tonnes in 2024 — a decrease of 10 902 tonnes, or roughly 14% [1][8].

- The drop reflects significant quota cuts: cod quotas were reduced by 34%, haddock by 43%, and the saithe quota in the northern zone by 6% [1][2].
- The report notes the quota reduction is to some extent offset by higher prices, which help the trawler fleet but challenge the land-based industry, which made another loss in 2024 [2].
- On outlook, the company states that earnings in the WildCatch segment have been significantly affected by the low quota basis and will remain so for some time, expecting 2025 and probably also 2026 to be challenging for the segment [3].
- It adds that quotas for the highest-value species, cod, will fall further in 2025, and that Lerøy must maintain efficient WildCatch operations while increasing capacity utilisation on land [4].
REQ: {"R1": "Catch volume was 64,991 tonnes (HOG) in 2024 versus 75,893 in 2023.", "R2": "The fall was about 10,900 tonnes (about 14%).", "R3": "Earnings are hit by the low quota basis and 2025 (probably 2026) are expected to be challenging."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "Quota cuts: cod –34%, haddock –43%, saithe –6%", "unit": null, "status": "unsupported"}, {"text": "Cod quotas will fall further in 2025 and Lerøy must maintain efficient WildCatch operations", "unit": null, "status": "extra"}]
strict reason: All required claims are present and accurate; extra details are unsupported or extra.
lax reason: All required claims correct; extra details do not alter core answer.

## 59. SETR_15500 V2-test-BOOK-fa-T1-002 (T1 fa) stricter=J2 structural=['claim_unsupported']
Q: In the first Majlis, how large was the annual budget deficit and what two remedies did the Majlis commission consider?
REF: Six crores, per Kasravi (elsewhere the same book gives six and a half crores a year). The commission (Vosough al-Dowleh was a member) considered two remedies: raising revenue and cutting outlays (Kasravi's word: 'daryaft').
ANS: در مجلس یکم، بودجهٔ سالانهٔ دولت شش کرور (و نیم) تومان کسری داشت و کمیسیون برای جبران آن دو راه در نظر گرفت: یکی افزودن به درآمد دولت و دیگری کاستن از دریافت‌ها (هزینه‌ها).

- درآمد سالانهٔ دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌آورد و باید با وام جبران می‌شد (گفتار چهارم، بندهای [3] و [4]).
- کمیسیون مالی که وثوق‌الدوله یکی از اعضای آن بود، دو راه را اندیشید: افزایش درآمد و کاهش دریافت‌ها (گفتار چهارم، بندهای [1] و [5]).
- برای افزایش درآمد، راه‌هایی مانند افزودن «تفاوت عمل» به درآمد، برانداختن تیول و برانداختن تسعیر در نظر گرفته شد (بندهای [5] و [17]).
- کمیسیون مالی از هزینه‌های سالانه هشت کرور کم کرد، از جمله کاستن از مستمری شاهزادگان و درباریان (گفتار هشتم، بندهای [2] و [12]).
REQ: {"R1": "The annual budget was six crores short (the book elsewhere gives six and a half).", "R2": "Remedies: increase revenue and reduce outlays."}
strict units: {'R1': 'correct', 'R2': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct'}
strict claims (non-correct): [{"text": "درآمد سالانهٔ دولت پانزده کرور تومان و نیم و دررفت آن بیست و یک کرور تومان و نیم بود که سالانه شش کرور کم می‌آورد", "unit": "R1", "status": "extra"}, {"text": "باید با وام جبران می‌شد", "unit": null, "status": "unsupported"}, {"text": "کمیسیون مالی که وثوق‌الدوله یکی از اعضای آن بود", "unit": null, "status": "extra"}, {"text": "برای افزایش درآمد، راه‌هایی مانند افزودن «تفاوت عمل» به درآمد، برانداختن تیول و برانداختن تسعیر در نظر گرفته شد", "unit": null, "status": "unsupported"}, {"text": "کمیسیون مالی از هزینه‌های سالانه هشت کرور کم کرد، از جمله کاستن از مستمری شاهزادگان و درباریان", "unit": null, "status": "unsupported"}, {"text": "(گفتار چهارم، بندهای [3] و [4])", "unit": null, "status": "non_substantive"}, {"text": "(گفتار چهارم، بندهای [1] و [5])", "unit": null, "status": "non_substantive"}, {"text": "(بندهای [5] و [17])", "unit": null, "status": "non_substantive"}, {"text": "(گفتار هشتم، بندهای [2] و [12])", "unit": null, "status": "non_substantive"}]
strict reason: Core claims match reference; extra/unsupported details do not contradict.
lax reason: Core claims match required facts; additional details are extra or duplicate, not affecting required answers.

## 60. SETR_15500 V2-test-BOOK-fa-T2-005 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: Per Bahar, under the northern oil concession law, what would happen if Standard Oil transferred the concession to another foreign government or company? And what did later negotiations reveal the company wanted regarding the Southern Oil Company?
REF: Under article 5 the company could not transfer the concession to a foreign government or company; if it did, the concession would be void. Later negotiations showed Standard Oil wanted to bring the Southern Oil Company into the concession or give it a 50% or 40% share of its stock - what the law forbade.
ANS: طبق مادهٔ پنجم امتیازنامه، در صورت تخلف و واگذاری امتیاز به دولت یا شرکت خارجی، امتیاز ملغی (فسخ) می‌شد؛ و روشن شد که استاندارد اویل می‌خواست کمپانی نفت جنوب (انگلیسی) را در امتیاز دخیل کند یا به‌اندازهٔ صد و پنجاه یا صد و چهل سهم در سهام خود شریکش سازد.

- مادهٔ ۵ امتیازنامه مقرر می‌داشت کمپانی حق واگذاری امتیاز به دولت یا شرکت خارجی را ندارد و در صورت تخلف امتیاز ملغی خواهد بود (شواهد ۱ و ۸).
- مجلس نیز انتقال امتیاز به دیگری یا مشارکت دادن دیگری را صریحاً منع کرده و موجب فسخ امتیاز می‌دانست (شواهد ۳ و ۶).
- مذاکرات بعدی نشان داد استاندارد اویل می‌خواهد کمپانی نفت جنوب را در امتیاز دخیل یا به‌قرار صد پنجاه یا صد چهل در اسهام خود شریک کند (شاهد ۲).
- این پیشنهاد از طرف دولت و کابینهٔ مشیرالدوله صریحاً رد شد (شاهد ۶).
REQ: {"R1": "Under article 5, transfer to a foreign government or company would void the concession.", "R2": "Standard Oil wanted to bring the Southern Oil Company into the concession or give it a 50% or 40% share.", "R3": "That demand ran against the law's transfer ban."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'missing'}
strict claims (non-correct): [{"text": "مادهٔ ۵ امتیازنامه مقرر می‌داشت کمپانی حق واگذاری امتیاز به دولت یا شرکت خارجی را ندارد و در صورت تخلف امتیاز ملغی خواهد بود", "unit": "R1", "status": "duplicate"}, {"text": "مجلس نیز انتقال امتیاز به دیگری یا مشارکت دادن دیگری را صریحاً منع کرده و موجب فسخ امتیاز می‌دانست", "unit": null, "status": "unsupported"}, {"text": "این پیشنهاد از طرف دولت و کابینهٔ مشیرالدوله صریحاً رد شد", "unit": null, "status": "unsupported"}, {"text": "(شواهد ۱ و ۸)", "unit": null, "status": "non_substantive"}, {"text": "(شواهد ۳ و ۶)", "unit": null, "status": "non_substantive"}, {"text": "(شاهد ۲)", "unit": null, "status": "non_substantive"}]
strict reason: Missing conclusion on conflict with law; unsupported claims about parliament and rejection.
lax reason: All required facts are present and accurate; extra statements do not conflict.

## 61. SETR_15500 V2-test-BOOK-fa-T3-002 (T3 fa) stricter=J1 structural=['claim_unsupported']
Q: The man elected first vice-president of the first Majlis: what was his role in the 1919 agreement, and per Britain's later claim, what sum had been shared between him and two other ministers?
REF: Vosough al-Dowleh. He concluded the 1919 agreement with Britain (subject to Majlis approval). Per Britain's claim, finance minister Sarem al-Dowleh received £130,000 at the signing, and this sum was shared between Vosough al-Dowleh, Sarem al-Dowleh and Nosrat al-Dowleh.
ANS: نایب‌رئیس یکم مجلس اول، میرزا حسن‌قلی خان وثوق‌الدوله بود و او بعدها نخست‌وزیر شد که قرارداد ۱۹۱۹ را با دولت بریتانیا منعقد کرد. طبق مطالبهٔ دولت بریتانیا، مبلغ صد و سی هزار لیره که در موقع انعقاد قرارداد ۱۹۱۹ توسط وزیر مالیه «صارم‌الدوله» اخذ شده بود، بین وثوق‌الدوله و صارم‌الدوله و نصرةالدوله تقسیم شده بود.

- وثوق‌الدوله نایب رئیس یکم مجلس اول بود (تاریخ مشروطه ایران، گفتار سوم).
- وثوق‌الدوله در سال دوم حکومت خود با تقاضاهای جدید مواجه شد که بصورت قرارداد به دولت ایران پیشنهاد گردید و در ذی‌القعده ۱۳۳۷ مطابق دسامبر ۱۹۱۹ قرارداد معروف بسته شد (تاریخ احزاب، بخش قرارداد ۱۹۱۹ و احمدشاه).
- مطالبهٔ دولت بریتانیا: صد و سی هزار لیره‌ای که در موقع انعقاد قرارداد ۱۹۱۹ توسط صارم‌الدوله اخذ و بین وثوق‌الدوله و صارم‌الدوله و نصرةالدوله تقسیم شده بود، جزء وجوهی بود که به‌مصرف امنیت جنوب رسیده و دولت ایران این مطالبه را نپذیرفت (تاریخ احزاب، بخش آغاز ضعف دولت قوام).
REQ: {"R1": "The first vice-president of the first Majlis was Vosough al-Dowleh.", "R2": "He concluded the 1919 agreement with Britain.", "R3": "£130,000, received by Sarem al-Dowleh, was shared between Vosough, Sarem and Nosrat al-Dowleh."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "وثوق‌الدوله نایب رئیس یکم مجلس اول بود (تاریخ مشروطه ایران، گفتار سوم)", "unit": "R1", "status": "duplicate"}, {"text": "وثوق‌الدوله در سال دوم حکومت خود با تقاضاهای جدید مواجه شد که بصورت قرارداد به دولت ایران پیشنهاد گردید", "unit": null, "status": "unsupported"}, {"text": "مطالبهٔ دولت بریتانیا: صد و سی هزار لیره‌ای که به‌مصرف امنیت جنوب رسیده و دولت ایران این مطالبه را نپذیرفت", "unit": "R3", "status": "extra"}]
strict reason: All required claims are present and accurate; extra and unsupported statements are flagged accordingly.
lax reason: All required claims correctly stated without contradictions or unsupported details.

## 62. SETR_15500 V2-test-LAW-fa-T1-002 (T1 fa) stricter=J1 structural=['req_missing']
Q: Under the Constitution, how many seats in the Islamic Consultative Assembly do the religious minorities have in total, and how are they divided?
REF: Five: one each for Zoroastrians and Jews, one jointly for Assyrian and Chaldean Christians, and one each for Armenian Christians of the south and of the north (Art. 64).
ANS: در مجموع پنج نماینده:
1. زرتشتیان: یک نماینده
2. کلیمیان: یک نماینده
3. مسیحیان آشوری و کلدانی: مجموعاً یک نماینده
4. مسیحیان ارمنی جنوب: یک نماینده
5. مسیحیان ارمنی شمال: یک نماینده
REQ: {"R1": "Zoroastrians elect one representative.", "R2": "Jews elect one representative.", "R3": "Assyrian and Chaldean Christians together elect one representative.", "R4": "Armenian Christians of the south and of the north elect one representative each.", "R5": "In total the religious minorities have five representatives."}
strict units: {'R1': 'correct', 'R2': 'missing', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'correct', 'R5': 'correct'}
strict claims (non-correct): [{"text": "کلیمیان: یک نماینده", "unit": null, "status": "extra"}]
strict reason: Claims match all required items except Jews; extra claim for کلیمیان; total seats correctly stated.
lax reason: All required claims present and accurate.

## 63. SETR_15500 V2-test-LAW-fa-T2-004 (T2 fa) stricter=J2 structural=['claim_unsupported']
Q: A barber's permanent licence has expired; the union sent a notice but the owner did not renew, and he is advertising on Instagram. Under the Guild System Law as amended in 1392, what is the shop's status and what fine does the advertising carry?
REF: After the one-month renewal notice, a unit whose licence is not renewed is treated as unlicensed (Art. 5 note 2). Advertising in any way by a guild member without a valid licence is prohibited, with a fine of 1 million to 250 million rials (Art. 17 note 3).
ANS: پاسخ: با انقضای اعتبار پروانه و عدم تمدید، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود و انتشار آگهی تبلیغاتی توسط فرد فاقد پروانه معتبر ممنوع است؛ متخلف به جریمه نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال محکوم می‌شود.

مبانی:
- طبق تبصره(2) ماده(5)، اتحادیه مکلف است با انقضای اعتبار پروانه، اخطاریه یک‌ماهه برای تبدیل پروانه موقت به دائم یا تمدید پروانه دائم صادر کند و در صورت عدم تبدیل یا تمدید، واحد صنفی در حکم واحد بدون پروانه تلقی می‌شود (مدت اعتبار پروانه دائم نیز طبق تبصره(1) ماده(5) پنج‌سال است).
- طبق تبصره(3) ماده(17)، انتشار هرگونه آگهی تبلیغاتی به هر طریق توسط فرد صنفی فاقد پروانه کسب معتبر ممنوع است و جریمه نقدی از یک میلیون (1.000.000) ریال تا دویست و پنجاه میلیون (250.000.000) ریال دارد.
- همچنین رسانه‌های گروهی و مؤسسات تولید محصولات چندرسانه‌ای مکلفند قبل از قبول سفارش آگهی، یک نسخه از پروانه کسب متقاضی را مطالبه کنند و در غیر این صورت به همان جریمه محکوم می‌شوند (تبصره(3) ماده(17)).
REQ: {"R1": "A unit whose licence is not renewed after the one-month notice is treated as unlicensed.", "R2": "Advertising by a guild member without a valid licence is prohibited.", "R3": "The fine is 1,000,000 to 250,000,000 rials."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "مدت اعتبار پروانه دائم طبق تبصره(1) ماده(5) پنج‌سال است", "unit": null, "status": "extra"}, {"text": "رسانه‌های گروهی و مؤسسات تولید محصولات چندرسانه‌ای مکلفند قبل از قبول سفارش آگهی، یک نسخه از پروانه کسب متقاضی را مطالبه کنند و در غیر این صورت به همان جریمه محکوم می‌شوند", "unit": null, "status": "unsupported"}]
strict reason: All required claims correct; extra and unsupported details present.
lax reason: All required claims match the packet; one additional non‑required claim is present.

## 64. SETR_15500 V2-test-LAW-fa-T3-001 (T3 fa) stricter=J2 structural=['claim_unsupported']
Q: Which official must initial a merchant's books before anything is written in them, and which book is exempt? Compare the 1303 Commercial Code with the current (1311) one.
REF: In both, all books of Art. 6 except the copy book must be initialled before use; the copy book needs only numbered pages. The signatory differs: 1303 - the president of the commercial court, another judge, a prosecution officer or the local governor (Art. 11); 1311 - a representative of the registration office (Art. 11).
ANS: قانون تجارت ۱۳۰۳ - ماده ۱۱: رئیس محکمه تجارت یا یکی از قضات دیگر عدلیه یا صاحب‌منصبان پارکه یا حاکم محل؛ دفتر کپیه از امضا معاف است ولی اوراق آن باید نمره ترتیبی داشته باشد.
قانون تجارت فعلی (۱۳۱۱) - ماده ۱۱: نماینده اداره‌ی ثبت (که مطابق نظام‌نامه‌ی وزارت عدلیه معین می‌شود)؛ دفتر کپیه از امضا معاف است ولی اوراق آن باید نمره‌ی ترتیبی داشته باشد.

مقایسه: در هر دو قانون دفتر کپیه از امضا معاف است، اما مقام امضاکننده تغییر کرده است: در قانون ۱۳۰۳ رئیس محکمه تجارت یا قضات عدلیه/صاحب‌منصبان پارکه/حاکم محل امضا می‌کردند، در حالی که در قانون فعلی این وظیفه بر عهده نماینده اداره‌ی ثبت است و حق امضا نیز هر صد صفحه یا کسور آن دو ریال تعیین شده است.
REQ: {"R1": "In both codes the copy book is exempt from initialling (numbered pages only).", "R2": "Under 1303 the books are initialled by the president of the commercial court, another judge, a prosecution officer or the local governor.", "R3": "Under 1311 the books are initialled by a representative of the registration office."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct'}
strict claims (non-correct): [{"text": "در قانون تجارت فعلی (۱۳۱۱) دفتر کپیه از امضا معاف است ولی اوراق آن باید نمره‌ی ترتیبی داشته باشد.", "unit": "R1", "status": "duplicate"}, {"text": "حق امضا نیز هر صد صفحه یا کسور آن دو ریال تعیین شده است.", "unit": null, "status": "unsupported"}]
strict reason: All required claims correct; extra detail on fee unsupported.
lax reason: All required claims are present and accurate; no contradictions or missing items.

## 65. SETR_15500 V2-test-LAW-en-T3-001 (T3 en) stricter=J2 structural=['claim_unsupported']
Q: Both the Digital Services Act and the Digital Markets Act use a 45 million user figure. What exactly does each Act count, and what else does the DMA require alongside that figure?
REF: DSA: 45 million or more average monthly active recipients of the service in the Union is the threshold for very large online platforms and search engines (Art. 33(1)). DMA: 45 million monthly active end users established or located in the Union in the last financial year, together with at least 10,000 yearly active business users established in the Union, triggers the presumption that a core platform service is an important gateway (Art. 3(2)(b)).
ANS: Digital Services Act - 45 million: average monthly active recipients of the service in the Union, calculated as an average over a period of six months, equivalent to 10 % of the Union population (Article 33(1), recital).
Digital Markets Act - 45 million: monthly active end users established or located in the Union in the last financial year, identified and calculated in accordance with the methodology and indicators set out in the Annex (Article 3(2)(b)).
Digital Markets Act - additional requirement: alongside the 45 million monthly active end users, the DMA requires at least 10 000 yearly active business users established in the Union, and that the thresholds in point (b) were met in each of the last three financial years (Article 3(2)(b)–(c)).

The DSA counts average monthly active recipients over six months as a proxy for societal reach (10 % of the Union population), while the DMA counts monthly active end users in the last financial year and additionally requires 10 000 yearly active business users and persistence of the thresholds over the last three financial years.
REQ: {"R1": "DSA counts average monthly active recipients of the service in the Union (>= 45 million).", "R2": "DMA counts monthly active end users established or located in the Union in the last financial year.", "R3": "DMA also requires at least 10,000 yearly active business users established in the Union.", "R4": "The DSA figure designates very large online platforms/search engines; the DMA figure triggers the gatekeeper 'important gateway' presumption."}
strict units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'} | lax units: {'R1': 'correct', 'R2': 'correct', 'R3': 'correct', 'R4': 'missing'}
strict claims (non-correct): [{"text": "DSA 45 million equivalent to 10% of the Union population", "unit": null, "status": "extra"}, {"text": "DMA requires persistence of the thresholds in point (b) over the last three financial years", "unit": null, "status": "unsupported"}, {"text": "DSA counts average monthly active recipients over six months as a proxy for societal reach (10% of the Union population)", "unit": "R1", "status": "duplicate"}, {"text": "DMA counts monthly active end users in the last financial year and additionally requires 10,000 yearly active business users and persistence of the thresholds over the last three financial years", "unit": null, "status": "unsupported"}]
strict reason: DMA's three-year persistence unsupported; R4 legal consequence missing.
lax reason: Core requirements are present; added details are not in the reference packet.
