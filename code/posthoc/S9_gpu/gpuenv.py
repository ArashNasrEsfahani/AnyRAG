"""S9 local-compute rules (author's): 2 BLAS/OMP threads, below-normal priority, one GPU job at a time.
Import this FIRST (before numpy/torch)."""
import ctypes
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
CODE = r"D:\KareOne\RAG-Paper-upgrades"
if CODE not in sys.path:
    sys.path.insert(0, CODE)
os.environ.setdefault("HF_HOME", r"D:\KareOne\RAG Paper\work\models\hf")


def below_normal():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS
        return True
    except Exception:  # noqa: BLE001
        return False


BELOW = below_normal()


def gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def nvsmi():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
