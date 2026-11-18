from reg import upsert
from conv_law_en import EU, UK
BS=chr(92)
for c,(slug,t,y) in EU.items():
    stem="eurlex-%s_%s_en"%(c,slug)
    upsert({"lane":"LAW_en","company":"eu","title_en":t,"year_gregorian":y,"language":"en","source_url":"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:%s"%c,
            "landing_url":"https://eur-lex.europa.eu/eli/reg/%s/%d/oj"%(y,int(c[6:])),"local_path":"law_en"+BS+stem+".html",
            "notes":"EUR-Lex Official Journal HTML (as adopted, not consolidated), CELEX %s, retrieved 2026-09-25; original law_en/_orig/eurlex_%s.html; converted to simple HTML (chapter/section/article headings, recitals and points as paragraphs)"%(c,c)},
           {"edition":"Official Journal text as adopted (original act, not a consolidated version)","published":y,"url":"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:%s"%c,
            "originals":["law_en/_orig/eurlex_%s.html"%c],"completeness":"complete act incl. recitals and annexes",
            "reuse_basis":"EUR-Lex content reusable under Commission Decision 2011/833/EU (reuse permitted with source acknowledgement); EU legislation",
            "converter":"scratch conv_law_en.py eu()"})
for (y,n),(slug,t) in UK.items():
    stem="ukpga-%d-%d_%s_en"%(y,n,slug)
    upsert({"lane":"LAW_en","company":"uk","title_en":t,"year_gregorian":str(y),"language":"en","source_url":"https://www.legislation.gov.uk/ukpga/%d/%d/data.xht?view=snippet&wrap=true"%(y,n),
            "landing_url":"https://www.legislation.gov.uk/ukpga/%d/%d"%(y,n),"local_path":"law_en"+BS+stem+".html",
            "notes":"legislation.gov.uk latest revised version as of retrieval 2026-09-25 (point in time = retrieval date), Open Government Licence v3.0; original law_en/_orig/legislation-gov-uk_ukpga-%d-%d.xht; converted to simple HTML; editorial annotations (F-notes, commencement/extent markers) removed, amended text kept"%(y,n)},
           {"edition":"latest revised version on legislation.gov.uk at retrieval (2026-09-25)","published":str(y),"url":"https://www.legislation.gov.uk/ukpga/%d/%d"%(y,n),
            "originals":["law_en/_orig/legislation-gov-uk_ukpga-%d-%d.xht"%(y,n)],"completeness":"complete act incl. schedules; editorial annotations dropped",
            "reuse_basis":"Open Government Licence v3.0 (Crown copyright), UK","converter":"scratch conv_law_en.py uk()"})
print("ok")
