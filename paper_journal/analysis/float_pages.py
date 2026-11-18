import re
aux = open('main.aux', encoding='utf-8', errors='ignore').read()
labs = re.findall(r'\\newlabel\{((?:tab|fig|sec|app)[^}]*)\}\{\{([^}]*)\}\{(\d+)\}', aux)
for kind in ('sec', 'flt'):
    rows = [(int(p), n, l) for l, n, p in labs if (l.startswith(('sec', 'app')) if kind == 'sec' else l.startswith(('tab', 'fig')))]
    for p, n, l in sorted(rows):
        print(kind.upper(), p, n, l)
    print()
log = open('main.log', encoding='utf-8', errors='ignore').read()
m = re.search(r'Output written on main.pdf \((\d+) pages', log)
print('pages', m.group(1) if m else '?')
