import json,collections
c=collections.defaultdict(lambda:[0,0.0])
for l in open('ledger.jsonl',encoding='utf-8'):
  r=json.loads(l); k=(r.get('approach','').split('|')[-1], r.get('purpose'), r.get('model'))
  c[k][0]+=1; c[k][1]+=r.get('cost_usd') or 0
for k,v in sorted(c.items()): print(k,v[0],round(v[1],4))
print('total', round(sum(v[1] for v in c.values()),4))
