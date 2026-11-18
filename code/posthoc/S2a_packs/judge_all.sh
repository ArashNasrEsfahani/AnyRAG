#!/bin/sh
# one judge process per arm (parallel shards; resumable)
cd "/d/KareOne/RAG Paper/work/posthoc_2026-11/S2a_packs"
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8
for a in $ARMS; do
  "/d/KareOne/RAG Paper/.venv/Scripts/python.exe" judge.py --arms $a --shard $a > judge_$a.log 2>&1 &
done
wait
echo JUDGE_ALL_DONE
