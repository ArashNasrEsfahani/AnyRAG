import io
f = 'main.tex'
s = io.open(f, encoding='utf-8').read()
BS = chr(92)
if 'placeins' not in s:
    s = s.replace(BS + 'usepackage{booktabs}',
                  BS + 'usepackage{booktabs}\n' + BS + 'usepackage[section]{placeins}% keep floats inside their own section', 1)
if BS + 'FloatBarrier\n' + BS + 'bibliography' not in s:
    s = s.replace(BS + 'bibliography{refs}', BS + 'FloatBarrier\n' + BS + 'bibliography{refs}', 1)
if BS + 'appendix\n' + BS + 'FloatBarrier' not in s:
    s = s.replace(BS + 'appendix\n', BS + 'appendix\n' + BS + 'FloatBarrier\n', 1)
io.open(f, 'w', encoding='utf-8', newline='').write(s)
print('ok')
