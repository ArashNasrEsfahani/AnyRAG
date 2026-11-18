import io
f = 'main.tex'
s = io.open(f, encoding='utf-8').read()
BS = chr(92)
old = BS + 'appendix\n' + BS + 'FloatBarrier\n'
new = (BS + 'FloatBarrier\n' + BS + 'clearpage\n' + BS + 'appendix\n'
       + '% clearly named appendix, starting on a fresh page\n'
       + BS + 'onecolumn\n'
       + BS + 'begin{center}{' + BS + 'Large' + BS + 'bfseries Appendix}' + BS + 'end{center}\n'
       + BS + 'twocolumn\n' if False else
       BS + 'FloatBarrier\n' + BS + 'clearpage\n' + BS + 'appendix\n'
       + '% clearly named appendix, starting on a fresh page\n'
       + BS + 'twocolumn[' + BS + 'begin{center}{' + BS + 'Large' + BS + 'bfseries Appendix}' + BS + 'end{center}' + BS + 'vspace{1em}]\n')
assert old in s, 'appendix block not found'
s = s.replace(old, new, 1)
io.open(f, 'w', encoding='utf-8', newline='').write(s)
print('ok')
