#!/usr/bin/env python3
"""Check scan_project.py on four tiny synthetic projects built in a temporary folder.

  A  Persian legal corpus, langchain + Chroma, chunk_size=2000, no reranker
  B  English FAQ bot, FAISS + OpenAI embeddings, CPU only, streaming
  C  empty repository
  D  mixed Persian/English corpus, bge-m3 + BM25 + RRF + bge-reranker, CUDA in Docker

For each it checks the inferred slots, the detected stack, and the end-to-end recommendation and change
list produced through choose.py. It also pipes `scan_project.py --json` into `choose.py --json` once.
Run:  python test_scan.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_project as sp  # noqa: E402

FA_LAW = ("ماده ۱۰ - قراردادهای خصوصی نسبت به کسانی که آن را منعقد نموده‌اند در صورتی که مخالف صریح قانون "
          "نباشد نافذ است. ماده ۱۱ - هرگاه کسی مالی را به دیگری بفروشد، خریدار مالک آن می‌شود و فروشنده باید "
          "آن را تحویل دهد. ماده ۱۲ - دادگاه پس از رسیدگی به دعوا، حکم مقتضی صادر می‌کند و طرفین می‌توانند "
          "ظرف بیست روز از تاریخ ابلاغ به دادگاه تجدیدنظر شکایت کنند. ")
FA_REPORT = ("شرکت در سال گذشته درآمد خود را دو برابر کرد و شمار کارکنان به ۴۲۰ نفر رسید. هیئت‌مدیره پیشنهاد "
             "می‌کند سود هر سهم میان سهام‌داران تقسیم شود و برنامهٔ گسترش بازار در سال آینده ادامه یابد. ")
EN_FAQ = ("How do I reset my password? Open the account page, choose Security and click Reset password. "
          "You will receive an email with a link that is valid for one hour. If the email does not arrive, "
          "check the spam folder and make sure the address on the account is correct. ")
EN_REPORT = ("The company doubled its revenue in the last financial year and the number of employees grew to "
             "420. The board proposes a dividend for every share and the expansion plan continues next year. ")


def arabic_forms(s):
    return s.replace("ی", "ي").replace("ک", "ك")


def write(root, rel, text):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def fixture_a(root):
    for i in range(3):
        write(root, "data/laws/law_%d.txt" % i, FA_LAW * 12)
    for i in range(3, 5):
        write(root, "data/laws/law_%d.txt" % i, arabic_forms(FA_LAW) * 12)   # varied spelling
    write(root, "app/rag/config.py",
          "from langchain_text_splitters import RecursiveCharacterTextSplitter\n"
          "splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)\n")
    write(root, "app/retrieve.py",
          "from langchain_chroma import Chroma\n"
          "from langchain_huggingface import HuggingFaceEmbeddings\n"
          "from langchain_openai import ChatOpenAI\n"
          "emb = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')\n"
          "db = Chroma(persist_directory='db', embedding_function=emb)\n"
          "SYSTEM_PROMPT = 'به پرسش کاربر فقط بر پایهٔ متن قانون پاسخ بده.'\n"
          "def answer(question):\n"
          "    docs = db.similarity_search(question, k=4)\n"
          "    return ChatOpenAI(model='gpt-4o-mini').invoke(SYSTEM_PROMPT + str(docs))\n")
    write(root, "eval/questions.json", json.dumps([
        {"question": "مهلت شکایت از حکم دادگاه چند روز است؟"},
        {"question": "قرارداد خصوصی در چه صورتی نافذ است؟"},
        {"question": "پس از فروش مال، مالک آن کیست؟"},
        {"question": "فروشنده پس از فروش چه تکلیفی دارد؟"},
        {"question": "دادگاه پس از رسیدگی چه می‌کند؟"},
        {"question": "شکایت تجدیدنظر به کجا داده می‌شود؟"},
    ], ensure_ascii=False))
    write(root, "requirements.txt", "langchain\nlangchain-chroma\nchromadb\nlangchain-openai\n")


def fixture_b(root):
    for i in range(6):
        write(root, "content/faq/article_%d.md" % i, "# Account help %d\n\n" % i + EN_FAQ * 10)
    write(root, "README.md", "# HelpBot\n\nA customer support FAQ bot that answers from our help-centre articles.\n")
    write(root, "bot/ingest.py", "CHUNK_SIZE = 800\nOVERLAP = 100\n")
    write(root, "bot/search.py",
          "import faiss\nfrom openai import OpenAI\n"
          "client = OpenAI()\nTOP_K = 5\n"
          "def embed(texts):\n"
          "    return client.embeddings.create(model='text-embedding-3-small', input=texts)\n"
          "def search(index, q):\n"
          "    D, I = index.search(q, TOP_K)\n"
          "    return I\n"
          "def reply(messages):\n"
          "    return client.chat.completions.create(model='gpt-4o-mini', messages=messages, stream=True)\n")
    write(root, "requirements.txt", "openai\nfaiss-cpu\nfastapi\n")


def fixture_c(root):
    os.makedirs(os.path.join(root, ".git"), exist_ok=True)
    write(root, ".git/config", "[core]\n")


def fixture_d(root):
    for i in range(3):
        write(root, "corpus/fa/report_%d.txt" % i, FA_REPORT * 15)
        write(root, "corpus/en/report_%d.txt" % i, EN_REPORT * 15)
    write(root, "search/pipeline.py",
          "import torch\nfrom hazm import Normalizer\nfrom rank_bm25 import BM25Okapi\n"
          "from FlagEmbedding import BGEM3FlagModel, FlagReranker\n"
          "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
          "model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)\n"
          "reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)\n"
          "chunk_size = 1000\nchunk_overlap = 200\ntop_k = 8\n"
          "def reciprocal_rank_fusion(lists, k=60):\n"
          "    pass\n")
    write(root, "Dockerfile", "FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\nRUN pip install torch --index-url https://download.pytorch.org/whl/cu121\n")
    write(root, "node_modules/junk/index.js", "import chromadb from 'chromadb'\n")   # must be skipped
    write(root, ".venv/pyvenv.cfg", "home = x\n")
    write(root, ".venv/lib/qdrant.py", "import qdrant_client\n")                    # must be skipped


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def changes_by(rec):
    return {c["option"]: c for c in rec["changes"]}


@case
def test_persian_legal(root):
    fixture_a(root)
    s = sp.scan(root)
    sl = s["slots"]
    rec = sp.recommend(s, {})
    ch = changes_by(rec)
    r = rec["recommendation"]
    return [
        ("q2 = fa", sl["q2"]["value"] == "fa" and sl["q2"]["confidence"] == "high"),
        ("q8 = mixed (Arabic ي/ك found)", sl["q8"]["value"] == "mixed"),
        ("q3 = moderate (k=4 x 2000 x 0.81)", sl["q3"]["value"] == "moderate"),
        ("corpus 5 txt files", s["corpus"]["files"] == 5 and s["corpus"]["types"] == {"txt": 5}),
        ("stack: chroma, langchain, e5", "chroma" in s["stack"].get("vector_db", {}) and
         "langchain" in s["stack"].get("framework", {}) and "multilingual-e5" in s["stack"].get("embedding", {})),
        ("no reranker, no BM25", not s["stack"].get("reranker") and not s["stack"].get("bm25")),
        ("chunk_size=2000 at app/rag/config.py:2", any(e["value"] == 2000 and e["where"] == "app/rag/config.py:2"
                                                       for e in s["stack"]["chunk_size"])),
        ("rec: hybrid, 1,000 chunks, reranker, Persian normalisation",
         r["channels"] == "ch.hybrid" and r["chunking"] == "ck.1000" and r["reranking_or_selection"] == "rr.ce"
         and r["normalisation"] == "nm.fa_mixed"),
        ("change: chunk 2000 -> ~1,000 (Tentative) in config.py",
         "av.chunks2000" in ch and ch["av.chunks2000"]["where"] == "app/rag/config.py:2"
         and ch["av.chunks2000"]["confidence"].startswith("Tentative")),
        ("change: add reranker after the Chroma query in app/retrieve.py",
         ch.get("rr.ce", {}).get("where", "").startswith("app/retrieve.py:") and ch["rr.ce"]["status"] == "change"),
        ("change: add BM25 channel", ch.get("ch.hybrid", {}).get("status") == "change"),
        ("change: add Persian normalisation", ch.get("nm.fa_mixed", {}).get("status") == "change"),
    ]


@case
def test_english_faq(root):
    fixture_b(root)
    s = sp.scan(root)
    sl = s["slots"]
    rec = sp.recommend(s, {})
    ch = changes_by(rec)
    r = rec["recommendation"]
    return [
        ("q1 = t1 (FAQ wording)", sl["q1"]["value"] == "t1"),
        ("q2 = en", sl["q2"]["value"] == "en" and sl["q2"]["confidence"] in ("medium", "high")),
        ("q4 = few (streaming)", sl["q4"]["value"] == "few"),
        ("q5 = no (faiss-cpu)", sl["q5"]["value"] == "no"),
        ("q8 not asked", sl["q8"]["applies"] is False and "q8" not in s["ask"]),
        ("q3 = tight (5 x 800 x 0.81)", sl["q3"]["value"] == "tight"),
        ("stack: faiss + OpenAI embeddings, provider openai",
         "faiss" in s["stack"].get("vector_db", {}) and "OpenAI embeddings" in s["stack"].get("embedding", {})
         and "openai" in s["resources"].get("llm_provider", {})),
        ("README.md is not counted as corpus", s["corpus"]["files"] == 6),
        ("rec: hybrid, no reranker needed for single facts",
         r["channels"] == "ch.hybrid" and r["reranking_or_selection"] == "rr.none_t1" and r["normalisation"] == "nm.en"),
        ("change: add BM25 next to FAISS", ch.get("ch.hybrid", {}).get("status") == "change"),
        ("scope: OpenAI embedding not tested", ch.get("ex.embedding", {}).get("status") == "scope"),
    ]


@case
def test_empty(root):
    fixture_c(root)
    s = sp.scan(root)
    rec = sp.recommend(s, {})
    return [
        ("nothing inferred", s["answers"] == {}),
        ("asks q1..q7", s["ask"] == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]),
        (".git skipped", s["scanned"]["files_read"] == 0),
        ("no corpus, no stack", s["corpus"]["files"] == 0 and not any(
            s["stack"].get(k) for k in ("vector_db", "bm25", "embedding", "reranker"))),
        ("rec = all-unsure defaults", rec["recommendation"]["reranking_or_selection"] == "rr.ce"
         and rec["recommendation"]["channels"] == "ch.hybrid"),
        ("no code changes", rec["changes"] == []),
    ]


@case
def test_mixed_bge(root):
    fixture_d(root)
    s = sp.scan(root)
    sl = s["slots"]
    rec = sp.recommend(s, {"q1": "t2"})
    ch = changes_by(rec)
    st = s["stack"]
    return [
        ("q2 = both", sl["q2"]["value"] == "both"),
        ("q5 = yes, high (CUDA image + torch.cuda)", sl["q5"]["value"] == "yes" and sl["q5"]["confidence"] == "high"),
        ("q8 = clean", sl["q8"]["value"] == "clean"),
        ("corpus 3 fa + 3 en files", s["corpus"]["file_languages"] == {"fa": 3, "en": 3}),
        ("stack: bge-m3, bge-reranker, rank_bm25, RRF, hazm",
         "bge-m3" in st.get("embedding", {}) and "bge-reranker" in st.get("reranker", {}) and
         "rank_bm25" in st.get("bm25", {}) and st.get("fusion") and st.get("persian_normaliser")),
        ("node_modules and .venv skipped", "chroma" not in st.get("vector_db", {}) and "qdrant" not in st.get("vector_db", {})),
        ("rec: hybrid + cross-encoder", rec["recommendation"]["reranking_or_selection"] == "rr.ce"),
        ("keeps: channels, chunks, reranker, normalisation",
         all(ch.get(o, {}).get("status") == "keep" for o in ("ch.hybrid", "ck.1000", "rr.ce", "nm.fa"))),
    ]


def pipe_check(root):
    """scan_project.py --json  ->  choose.py --json  (as the skill tells the agent to do)."""
    fixture_a(root)
    out = subprocess.run([sys.executable, os.path.join(HERE, "scan_project.py"), root, "--json"],
                         capture_output=True, check=True)
    path = os.path.join(root, "..", os.path.basename(root) + "_scan.json")
    with open(path, "wb") as f:
        f.write(out.stdout)
    res = subprocess.run([sys.executable, os.path.join(HERE, "choose.py"), "--json", path, "--format", "json"],
                         capture_output=True, check=True)
    os.remove(path)
    rec = json.loads(res.stdout.decode("utf-8"))
    return [("choose.py reads scan JSON", rec["answers"]["q2"] == "fa" and rec["recommendation"]["chunking"] == "ck.1000")]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    total = failed = 0
    for fn in CASES + [pipe_check]:
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "proj")
            os.makedirs(root)
            checks = fn(root)
        print(fn.__name__)
        for name, ok in checks:
            total += 1
            failed += not ok
            print("  %s  %s" % ("PASS" if ok else "FAIL", name))
    print("\nScan checks: %d of %d passed" % (total - failed, total))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
