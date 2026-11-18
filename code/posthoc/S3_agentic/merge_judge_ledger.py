"""Append the judge calls (judge_multi's own ledgers under judge/) to ledger.jsonl once (skips call_ids already there)."""
import json, glob, os
seen = set()
for l in open('ledger.jsonl', encoding='utf-8'):
    c = json.loads(l)
    if c.get('purpose') == 'judge':
        seen.add(c.get('call_id'))
rows = []
for f in sorted(glob.glob(os.path.join('judge', '*', '*', 'judge_ledger_*.jsonl'))):
    parts = os.path.normpath(f).split(os.sep)
    arm, split = parts[1], parts[2]
    for l in open(f, encoding='utf-8'):
        c = json.loads(l)
        if c.get('kind') != 'chat' or c.get('call_id') in seen:
            continue
        rows.append(dict(ts=c.get('ts'), stream='S3_agentic', model=c.get('model'), purpose='judge', arm='JUDGE:' + arm,
                         run_tag='judge_' + split, question_id=c.get('question_id'), ok=c.get('ok'),
                         provider=c.get('provider'), provider_pin=c.get('provider_pin'),
                         prompt_tokens=c.get('prompt_tokens'), completion_tokens=c.get('completion_tokens'),
                         cost_usd=c.get('cost_usd') or 0.0, call_id=c.get('call_id'), source_ledger=f))
with open('ledger.jsonl', 'a', encoding='utf-8') as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows))
