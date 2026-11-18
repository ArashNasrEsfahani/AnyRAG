"""Print the word count of the abstract in sections/00_abstract.tex."""
s = open("sections/00_abstract.tex", encoding="utf-8").read()
body = s.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
print(len(body.split()), "words")
