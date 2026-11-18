from reg import upsert
B=[(1228,"darwin_origin-of-species-1st-ed","darwin","On the Origin of Species (1st edition, 1859)","1859","Charles Darwin; John Murray, London, 1st ed."),
(2009,"darwin_origin-of-species-6th-ed","darwin","On the Origin of Species (6th edition, 1872)","1872","Charles Darwin; 6th London ed. with additions and corrections"),
(34901,"mill_on-liberty","mill","On Liberty","1859","John Stuart Mill; Longman, Green 1869 printing"),
(4239,"malthus_essay-on-population","malthus","An Essay on the Principle of Population (1798)","1798","Thomas Malthus; 1st ed."),
(815,"tocqueville_democracy-in-america-part1","tocqueville","Democracy in America, Volume 1","1835","Alexis de Tocqueville; tr. Henry Reeve"),
(1015,"parkman_oregon-trail","parkman","The Oregon Trail","1849","Francis Parkman Jr."),
(20203,"franklin_autobiography","franklin","Autobiography of Benjamin Franklin","1791","Benjamin Franklin; ed. Frank Woodworth Pine 1916"),
(23,"douglass_narrative","douglass","Narrative of the Life of Frederick Douglass, an American Slave","1845","Frederick Douglass"),
(833,"veblen_theory-of-leisure-class","veblen","The Theory of the Leisure Class","1899","Thorstein Veblen"),
(14474,"faraday_chemical-history-of-a-candle","faraday","The Chemical History of a Candle","1861","Michael Faraday; ed. William Crookes"),
(147,"paine_common-sense","paine","Common Sense","1776","Thomas Paine")]
for i,slug,co,t,y,ed in B:
    stem="gutenberg-%d_%s_en"%(i,slug)
    upsert({"lane":"BOOK_en","company":co,"title_en":t,"year_gregorian":y,"language":"en","source_url":"https://www.gutenberg.org/cache/epub/%d/pg%d-images.html"%(i,i),
            "landing_url":"https://www.gutenberg.org/ebooks/%d"%i,"local_path":"book_en\%s.html"%stem,
            "notes":"Project Gutenberg eBook #%d (public domain in the USA; PG licence boilerplate stripped); original HTML in book_en/_orig/pg%d-images.html; converted to simple HTML (headings/paragraphs/tables; page numbers, TOC tables, illustrations, PG header/footer removed)"%(i,i)},
           {"edition":ed,"published":y,"url":"https://www.gutenberg.org/ebooks/%d"%i,"originals":["book_en/_orig/pg%d-images.html"%i],
            "completeness":"complete book text as published by Project Gutenberg","reuse_basis":"public domain (USA; author died >70 years ago, so PD in most jurisdictions); Project Gutenberg trademark licence boilerplate removed",
            "converter":"scratch conv_gut.py (stdlib DOM simplifier)"})
print("ok")
