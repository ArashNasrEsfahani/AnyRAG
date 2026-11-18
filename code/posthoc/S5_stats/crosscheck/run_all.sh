#!/bin/sh
# Runs every cross-check fit in parallel (single-threaded BLAS per process). CPU only.
cd "$(dirname "$0")"
PY="../../../../.venv/Scripts/python.exe"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
$PY crosscheck.py freq > freq.log 2>&1 &
for m in vb map; do for l in default diffuse; do for f in 0 1; do for r in HYBRID_6000 HR_6000 LEX5B_6000 SETR_6000; do
  echo "$m $l $f $r"; done; done; done; done | xargs -P 18 -L 1 sh -c "$PY crosscheck.py bayes \$0 \$1 \$2 \$3" > bayes.log 2>&1
wait
echo ALLDONE
