"""S12 shared helpers: size-specific configs (only chunking.chunk_chars / overlap_chars changed, plus index/lock paths
redirected into this folder so nothing shared is written), local-job settings."""
import ctypes
import os
import subprocess

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ.setdefault("HF_HOME", r"D:\KareOne\RAG Paper\work\models\hf")

import packs as PK  # noqa: E402  (frozen code on sys.path; NOTE packs sets CUDA_VISIBLE_DEVICES="" -> callers needing CUDA pop it first)

HERE = PK.HERE
SIZES = {500: 100, 1000: 200, 2000: 400}


def below_normal():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
        return True
    except Exception:  # noqa: BLE001
        return False


def cfg_for(size):
    if size == 1000:
        return PK.CFG                                  # the frozen config and the frozen index
    return PK.CFG.copy_with({"chunking": {"chunk_chars": size, "overlap_chars": SIZES[size]},
                             "paths": {"index_root": str(HERE / ("idx%d" % size)),
                                       "locks_root": str(HERE / ("locks%d" % size))}})


def nvsmi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


SETS = ["BOOK_en", "BOOK_fa", "C", "D_en", "LAW_en", "LAW_fa"]
