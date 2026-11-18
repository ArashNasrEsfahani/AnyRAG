#!/usr/bin/env python3
"""choose-rag: read a project folder and infer the answers to the eight chooser questions.

Read-only and local. Python 3.8+ standard library only. It never executes project code, never
imports project modules, never opens a network connection and never writes anywhere: it walks the
folder, reads a capped sample of files as bytes and matches text patterns.

What it infers (each with a confidence: high, medium, low or unknown, and file:line evidence):

  corpus      document folders and file types (pdf, md, html, docx, txt, json), count, size,
              a rough number of characters, and the script mix of a text sample
  query side  the language of test questions, UI strings and prompt templates, where found
  stack       vector DBs, BM25, embedding models, rerankers, frameworks, fusion, query
              rewriting, agents, Persian normalisation, chunk_size / overlap, top-k, budgets
  resources   GPU hints, LLM providers, latency hints
  q1..q8      the chooser slots (references/chooser.json option ids), ready for choose.py

Usage
  python scan_project.py <path>                       # slot table + profile
  python scan_project.py <path> --json                # machine-readable; "answers" feeds choose.py
  python scan_project.py <path> --recommend           # + choose.py's recommendation and the
                                                      #   changes to make in this project's files
  python scan_project.py <path> --recommend --q3 moderate --q4 few   # with confirmed answers

  python scan_project.py <path> --json > scan.json
  python choose.py --json scan.json --q4 few           # choose.py reads scan.json["answers"]
"""
import argparse
import html
import json
import os
import re
import sys
import zipfile
import zlib
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import choose  # noqa: E402  (the skill's own rules script, not project code)

VERSION = "2026-11-18"

LIMITS = {
    "max_entries": 60000,          # files looked at (stat only) during the walk
    "max_files_read": 2500,        # files whose bytes are read
    "max_file_bytes": 256 * 1024,  # bytes read from one text or code file
    "max_pdf_bytes": 4 * 1024 * 1024,
    "max_docx_xml_bytes": 4 * 1024 * 1024,
    "max_total_bytes": 64 * 1024 * 1024,
    "max_corpus_samples": 300,     # corpus files sampled for language
    "sample_chars": 40000,         # characters kept per sampled file
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bower_components", "venv", ".venv", "env", ".env",
    "virtualenv", "__pycache__", "build", "dist", "target", "out", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages", ".next", ".nuxt", ".svelte-kit", ".idea",
    ".vscode", "coverage", "htmlcov", ".cache", ".gradle", ".terraform", "bin", "obj",
    ".ipynb_checkpoints", "wandb", "mlruns", ".eggs", "vendor",
}
HIDDEN_OK = {".github", ".streamlit", ".devcontainer"}

BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff", ".svg", ".psd",
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".whl", ".egg", ".jar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib", ".pyc", ".pyo", ".class",
    ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".gguf", ".h5", ".pkl", ".pickle",
    ".npy", ".npz", ".parquet", ".arrow", ".feather", ".db", ".sqlite", ".sqlite3", ".faiss",
    ".index", ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".mov", ".avi", ".mkv", ".webm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".xlsx", ".xls", ".pptx", ".ppt", ".doc",
    ".odt", ".ods", ".epub", ".lock",
}
DOC_EXT = {".pdf": "pdf", ".md": "md", ".markdown": "md", ".html": "html", ".htm": "html",
           ".docx": "docx", ".txt": "txt", ".json": "json", ".jsonl": "json"}
CODE_EXT = {".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
            ".go", ".java", ".kt", ".rb", ".rs", ".cs", ".php", ".scala", ".swift",
            ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".sql", ".r", ".jl"}
CONFIG_NAMES = re.compile(r"^(requirements.*\.txt|constraints.*\.txt|dockerfile.*|docker-compose.*|compose\.ya?ml|"
                          r"pipfile|pyproject\.toml|setup\.py|setup\.cfg|package\.json|environment\.ya?ml|"
                          r"makefile|\.env\.(example|sample|template)|.*\.env\.example)$", re.I)
UI_EXT = {".html", ".htm", ".jsx", ".tsx", ".vue", ".svelte"}

CORPUS_DIRS = {
    "data", "dataset", "datasets", "corpus", "corpora", "documents", "document", "docstore", "kb",
    "knowledge", "knowledge_base", "knowledgebase", "content", "contents", "files", "pdfs", "pdf",
    "raw", "sources", "source_docs", "library", "texts", "text", "articles", "faq", "faqs", "laws",
    "law", "statutes", "contracts", "reports", "uploads", "ingest", "inputs", "input", "books",
    "manuals", "policies", "papers", "pages", "wiki", "kb_docs", "docs_corpus", "collection",
}
NOT_CORPUS_DIRS = {"results", "result", "output", "outputs", "logs", "log", "runs", "cache", "tmp",
                   "checkpoints", "models", "node_modules", "static", "public", "assets", "locales",
                   "locale", "i18n", "translations", "lang", "fixtures_out", "snapshots"}
NOT_CORPUS_NAME = re.compile(r"^(readme|changelog|changes|license|licence|contributing|code_of_conduct|"
                             r"security|authors|notice|todo|history|package(-lock)?|tsconfig.*|jsconfig|"
                             r"composer|manifest|settings|launch|tasks|\.?eslintrc.*|\.?prettierrc.*|"
                             r"babel\.config|vercel|netlify|app|appsettings.*|config.*|.*\.config)$", re.I)
QUERY_NAME = re.compile(r"(question|queries|query|_qa\b|^qa\b|qa_|q&a|eval|testset|test_set|golden|"
                        r"benchmark|prompts?\b|sample_questions|examples?_questions)", re.I)
QUERY_DIRS = {"questions", "queries", "qa", "eval", "evals", "evaluation", "testset", "testsets", "golden"}
LOCALE_DIRS = {"locales", "locale", "i18n", "translations", "lang", "langs", "messages"}

CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}

# ---------------------------------------------------------------- patterns (applied to code/config)
P = re.compile
STACK_PATTERNS = OrderedDict([
    ("vector_db", [
        ("faiss", P(r"\bfaiss\b", re.I)),
        ("chroma", P(r"\bchromadb\b|\bChroma\b|langchain[_-]chroma")),
        ("qdrant", P(r"qdrant", re.I)),
        ("weaviate", P(r"weaviate", re.I)),
        ("pgvector", P(r"pgvector|PGVector")),
        ("milvus", P(r"milvus", re.I)),
        ("lancedb", P(r"lancedb", re.I)),
        ("elasticsearch", P(r"elasticsearch|ElasticsearchStore", re.I)),
        ("opensearch", P(r"opensearch", re.I)),
        ("pinecone", P(r"pinecone", re.I)),
    ]),
    ("bm25", [
        ("rank_bm25", P(r"rank[_-]bm25|BM25Okapi|BM25Plus\b")),
        ("bm25s", P(r"\bbm25s\b")),
        ("BM25Retriever", P(r"BM25Retriever|BM25Encoder")),
        ("elasticsearch", P(r"elasticsearch", re.I)),
        ("opensearch", P(r"opensearch", re.I)),
        ("pyserini", P(r"pyserini", re.I)),
        ("sqlite fts5", P(r"\bfts5\b", re.I)),
        ("bm25 (own code)", P(r"\bbm25\b", re.I)),
    ]),
    ("embedding", [
        ("bge-m3", P(r"bge[-_]m3|BGEM3", re.I)),
        ("multilingual-e5", P(r"multilingual[-_]e5|intfloat/|\be5[-_](small|base|large)", re.I)),
        ("OpenAI embeddings", P(r"text-embedding-(3|ada)|OpenAIEmbeddings|AzureOpenAIEmbeddings|embeddings\.create")),
        ("sentence-transformers", P(r"sentence[_-]transformers|SentenceTransformer\b|HuggingFaceEmbeddings")),
        ("other embedding model", P(r"all-MiniLM|all-mpnet|nomic-embed|jina-embeddings|\bgte-|bge-(small|base|large)|"
                                    r"voyage-|embed-(english|multilingual)|mxbai-embed|GoogleGenerativeAIEmbeddings|"
                                    r"CohereEmbeddings|OllamaEmbeddings", re.I)),
    ]),
    ("reranker", [
        ("bge-reranker", P(r"bge[-_]reranker|FlagReranker", re.I)),
        ("cohere rerank", P(r"CohereRerank|rerank-(english|multilingual|v3)|co\.rerank\(", re.I)),
        ("cross-encoder", P(r"CrossEncoder\b|cross-encoder/|cross_encoder|ms-marco-MiniLM", re.I)),
        ("other reranker", P(r"jina-reranker|colbert|FlashrankRerank|SentenceTransformerRerank|mxbai-rerank", re.I)),
    ]),
    ("llm_selection", [
        ("LLM reranking / selection", P(r"LLMRerank|RankGPT|LLMChainFilter|LLMListwiseRerank|LLMChainExtractor|"
                                        r"llm[_ ]select|select_passages|passage[_ ]selection|llm[_ ]rerank", re.I)),
    ]),
    ("framework", [
        ("langchain", P(r"\blangchain", re.I)),
        ("llama_index", P(r"llama[_-]index", re.I)),
        ("haystack", P(r"\bhaystack\b", re.I)),
        ("dspy", P(r"\bdspy\b", re.I)),
        ("langgraph", P(r"\blanggraph\b", re.I)),
        ("semantic-kernel", P(r"semantic[_-]kernel", re.I)),
    ]),
    ("fusion", [
        ("reciprocal rank fusion", P(r"reciprocal[_ ]rank|\brrf\b|EnsembleRetriever|QueryFusionRetriever|"
                                     r"hybrid[_ ]search|hybrid[_ ]retriev", re.I)),
    ]),
    ("query_rewriting", [
        ("HyDE / multi-query", P(r"\bhyde\b|HypotheticalDocument|hypothetical[_ ]document|MultiQueryRetriever|"
                                 r"multi[_-]query|query[_ ]rewrit|rewrite[_ ]query|step[_-]back[_ ]prompt", re.I)),
    ]),
    ("agent", [
        ("agent loop", P(r"AgentExecutor|create_react_agent|create_tool_calling_agent|ReActAgent|FunctionAgent|"
                         r"initialize_agent|\bagentic\b|search[_ ]agent|grep[_ ]agent|StateGraph\(", re.I)),
    ]),
    ("persian_normaliser", [
        ("Persian normalisation", P(r"\bhazm\b|parsivar|\\u064[aA]|\\u0643|['\"]\u064a['\"]|['\"]\u0643['\"]|"
                                    r"\\u200[cC]|persian[_ ]?normal|normali[sz]e_(fa|persian)", re.I)),
    ]),
    ("retrieval_call", [
        ("retrieval call", P(r"\.similarity_search\w*\(|\.as_retriever\(|get_relevant_documents\(|"
                             r"collection\.query\(|\bindex\.search\(|client\.search\(|\.query_points\(|"
                             r"max_marginal_relevance_search|\.retrieve\(|\.get_scores\(|knn_search|"
                             r"\.search\(\s*(q|query|question)")),
    ]),
    ("long_context", [
        ("whole documents in the prompt", P(r"chain_type\s*=\s*['\"]stuff['\"]|long[_ ]context|full[_ ]document[_ ]prompt", re.I)),
    ]),
    ("graph_or_sql", [
        ("graph / SQL retrieval", P(r"\bneo4j\b|graphrag|KnowledgeGraph|SQLDatabaseChain|text[_-]?to[_-]?sql|NL2SQL", re.I)),
    ]),
    ("chat_memory", [
        ("multi-turn chat memory", P(r"ConversationBufferMemory|chat_history|ChatMessageHistory|ConversationalRetrievalChain", re.I)),
    ]),
])

RESOURCE_PATTERNS = OrderedDict([
    ("gpu_infra", [
        ("CUDA wheels (cu1xx)", P(r"\+cu1\d\d|/cu1\d\d\b|whl/cu1\d\d|cuda1\d", re.I)),
        ("CUDA base image", P(r"nvidia/cuda|pytorch/pytorch:.*cuda", re.I)),
        ("GPU in Docker", P(r"--gpus|runtime:\s*nvidia|capabilities:\s*\[\s*gpu|driver:\s*nvidia", re.I)),
        ("GPU packages", P(r"faiss-gpu|onnxruntime-gpu|\bcupy\b|^nvidia-[a-z]", re.I)),
        ("vLLM (GPU server)", P(r"\bvllm\b", re.I)),
    ]),
    ("gpu_code", [
        ("torch.cuda", P(r"torch\.cuda")),
        ("device cuda", P(r"device\s*[=:]\s*['\"]cuda|\.to\(\s*['\"]cuda|\.cuda\(\)|CUDA_VISIBLE_DEVICES|use_fp16\s*=\s*True|nvidia-smi")),
    ]),
    ("cpu_only", [
        ("CPU-only packages", P(r"faiss-cpu|download\.pytorch\.org/whl/cpu|\+cpu\b", re.I)),
        ("device cpu", P(r"device\s*[=:]\s*['\"]cpu['\"]")),
    ]),
    ("local_models", [
        ("local model libraries", P(r"\btorch\b|\btransformers\b|sentence[_-]transformers|FlagEmbedding|HuggingFaceEmbeddings|langchain_huggingface|"
                                    r"llama[_-]cpp|\bctransformers\b", re.I)),
    ]),
    ("llm_provider", [
        ("openai", P(r"\bfrom openai\b|\bimport openai\b|ChatOpenAI|api\.openai\.com|\bopenai\s*[>=<~]|['\"]gpt-[345o]")),
        ("anthropic", P(r"\banthropic\b|ChatAnthropic|['\"]claude-", re.I)),
        ("openrouter", P(r"openrouter", re.I)),
        ("ollama", P(r"ollama", re.I)),
        ("vllm", P(r"\bvllm\b", re.I)),
        ("azure openai", P(r"AzureOpenAI|AzureChatOpenAI")),
        ("google gemini", P(r"google\.generativeai|google-genai|google\.genai|ChatGoogleGenerativeAI|['\"]gemini-", re.I)),
        ("groq", P(r"\bgroq\b", re.I)),
        ("mistral", P(r"mistralai", re.I)),
        ("cohere", P(r"\bimport cohere\b|\bcohere\.Client|ChatCohere")),
        ("together", P(r"together\.ai|\bfrom together\b|api\.together", re.I)),
        ("local transformers", P(r"AutoModelForCausalLM|pipeline\(\s*['\"]text-generation")),
        ("llama.cpp", P(r"llama[_-]cpp", re.I)),
    ]),
    ("latency_fast", [
        ("search-as-you-type", P(r"autocomplete|typeahead|type[-_ ]ahead|search[-_ ]as[-_ ]you[-_ ]type|instant[-_ ]search|debounce", re.I)),
        ("real-time endpoint", P(r"websocket|real[-_ ]?time[-_ ](api|endpoint|search|answer|chat|response)|"
                                 r"latency[_ ](budget|slo|target)\s*[=:]", re.I)),
    ]),
    ("latency_stream", [
        ("streaming", P(r"stream\s*=\s*True|streaming\s*=\s*True|StreamingResponse|EventSourceResponse|"
                        r"text/event-stream|\.astream\(|\.stream\(")),
        ("chat UI", P(r"gradio|streamlit|chainlit|ChatInterface|st\.chat_", re.I)),
    ]),
    ("latency_batch", [
        ("batch / offline", P(r"\bbatch[_ ](job|mode|process|inference)|\boffline\b|nightly|\bcron\b|celery", re.I)),
    ]),
])
TIMEOUT_RE = P(r"\btimeout\s*[=:]\s*(\d+(?:\.\d+)?)\b")

CHUNK_RE = P(r"\b(chunk[_ ]?size|chunk_chars|chunk_len(?:gth)?|max_chunk(?:_size|_chars|_tokens)?|chunk_tokens|"
             r"CHUNK_SIZE|CHUNK_CHARS|CHUNK_TOKENS)\b['\"]?\s*[:=]\s*(\d[\d_]*)", re.I)
OVERLAP_RE = P(r"\b(chunk[_ ]?overlap|overlap(?:_chars|_tokens)?|CHUNK_OVERLAP)\b['\"]?\s*[:=]\s*(\d[\d_]*)", re.I)
TOPK_RE = P(r"\b(top[_-]?k|similarity_top_k|n_results|num_results|top_n|topn|n_docs|num_docs|search_k|"
            r"k_retrieve|retrieve_k|TOP_K|k|limit)\b['\"]?\s*[:=]\s*(\d+)\b", re.I)
TOPK_CONTEXT = P(r"search|retriev|query|top|rerank|similar|n_results|fetch", re.I)
BUDGET_RE = P(r"\b(\w*(?:context|evidence|pack|prompt)\w*_(?:budget|chars|characters|tokens|limit|max|window)\w*|"
              r"(?:context|evidence|pack|char|token)_?budget\w*|budget(?:_chars|_tokens)?|max_context\w*|"
              r"max_(?:context|evidence|pack)_\w+)\b['\"]?\s*[:=]\s*(\d[\d_,]*)", re.I)
TOKEN_SPLITTER = P(r"TokenTextSplitter|from_tiktoken_encoder|SentenceSplitter|TokenSplitter|"
                   r"SentenceTransformersTokenTextSplitter|llama_index", re.I)

Q1_SINGLE = P(r"\bfaqs?\b|frequently asked|help ?desk|customer (support|service)|support (bot|ticket|agent|article)|"
              r"knowledge ?base article|\u0633\u0648\u0627\u0644\u0627\u062a \u0645\u062a\u062f\u0627\u0648\u0644|"
              r"\u067e\u0631\u0633\u0634(\u200c| )?\u0647\u0627\u06cc \u0645\u062a\u062f\u0627\u0648\u0644", re.I)
Q1_CROSS = P(r"\bcompar(e|es|ing|ison)\b|across (documents|reports|files|sources|years|contracts)|multi[- ]?hop|"
             r"multi[- ]document|cross[- ]document|difference between|summari[sz]e across|\bversus\b|\bvs\.?\s|"
             r"trend|\u0645\u0642\u0627\u06cc\u0633\u0647|\u062a\u0641\u0627\u0648\u062a|\u062a\u062d\u0644\u06cc\u0644", re.I)
Q1_WITHIN = P(r"\banaly[sz](e|is)\b|clause|section|chapter|\u0641\u0635\u0644|\u0645\u0627\u062f\u0647", re.I)
Q7_PRIVATE = P(r"internal|confidential|private|proprietary|polic(y|ies)|handbook|employee|\bhr\b|contract|invoice|"
               r"ticket|customer|annual[_ -]report|financial[_ -]statement|\u0645\u062d\u0631\u0645\u0627\u0646\u0647|"
               r"\u062f\u0627\u062e\u0644\u06cc", re.I)
Q7_PUBLIC = P(r"wikipedia|gutenberg|constitution|civil[ _-]code|\bgdpr\b|eu[ _-]regulation|bible|quran|"
              r"public[ _-]domain|arxiv|\u0642\u0627\u0646\u0648\u0646[ _]\u0627\u0633\u0627\u0633\u06cc|"
              r"\u0642\u0627\u0646\u0648\u0646[ _]\u0645\u062f\u0646\u06cc|\u0634\u0627\u0647\u0646\u0627\u0645\u0647", re.I)

STRING_LIT = P(r"(?:[rbuf]{0,2})(\"\"\"|'''|\"|'|`)((?:\\.|(?!\1).){8,400}?)\1")
PROMPTISH = P(r"prompt|system|template|question|placeholder|label|title|message|instruction|ask|query|greeting", re.I)

# ---------------------------------------------------------------- scripts and languages
PERSIAN_ONLY = set("\u067e\u0686\u0698\u06af")          # \u067e \u0686 \u0698 \u06af
PERSIAN_YK = set("\u06cc\u06a9")                          # \u06cc \u06a9
ARABIC_YK = set("\u064a\u0643")                           # \u064a \u0643
ARABIC_ONLY = set("\u0629\u0649")                         # \u0629 \u0649
EN_STOP = {"the", "and", "of", "to", "in", "is", "that", "for", "with", "are", "on", "as", "this", "be", "by", "what", "how"}
WORD_RE = P(r"[A-Za-z]+")


def script_of(ch):
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        return "arabic_script" if ch.isalpha() else None
    if ch.isalpha():
        if o < 0x250 or 0x1E00 <= o <= 0x1EFF:
            return "latin"
        if 0x0400 <= o <= 0x04FF:
            return "cyrillic"
        if 0x0590 <= o <= 0x05FF:
            return "hebrew"
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if 0x0370 <= o <= 0x03FF:
            return "greek"
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
            return "cjk"
        return "other"
    return None


class TextStats:
    """Letter counts by script plus the Persian orthography counters needed for q8."""

    def __init__(self):
        self.scripts = Counter()
        self.persian_only = self.persian_yk = self.arabic_yk = self.arabic_only = 0
        self.zwnj = self.mi_spaced = self.mi_joined = 0
        self.digits = Counter()
        self.latin_words = self.en_stop = 0

    def add(self, text):
        for ch in text:
            s = script_of(ch)
            if s:
                self.scripts[s] += 1
                if s == "arabic_script":
                    if ch in PERSIAN_ONLY:
                        self.persian_only += 1
                    elif ch in PERSIAN_YK:
                        self.persian_yk += 1
                    elif ch in ARABIC_YK:
                        self.arabic_yk += 1
                    elif ch in ARABIC_ONLY:
                        self.arabic_only += 1
            elif ch == "\u200c":
                self.zwnj += 1
            elif "0" <= ch <= "9":
                self.digits["ascii"] += 1
            elif "\u06f0" <= ch <= "\u06f9":
                self.digits["persian"] += 1
            elif "\u0660" <= ch <= "\u0669":
                self.digits["arabic_indic"] += 1
        self.mi_spaced += len(re.findall(r"(?:^|\s)(?:\u0646?\u0645\u06cc|\u0646?\u0645\u064a) (?=[\u0600-\u06ff])", text))
        self.mi_joined += len(re.findall(r"(?:^|\s)\u0646?\u0645[\u06cc\u064a]\u200c", text))
        words = WORD_RE.findall(text[:20000])
        self.latin_words += len(words)
        self.en_stop += sum(1 for w in words if w.lower() in EN_STOP)

    def merge(self, other):
        self.scripts.update(other.scripts)
        self.digits.update(other.digits)
        for a in ("persian_only", "persian_yk", "arabic_yk", "arabic_only", "zwnj", "mi_spaced",
                  "mi_joined", "latin_words", "en_stop"):
            setattr(self, a, getattr(self, a) + getattr(other, a))

    @property
    def letters(self):
        return sum(self.scripts.values())

    def arabic_kind(self):
        """'fa' or 'ar' for the Arabic-script letters."""
        fa = self.persian_only + self.persian_yk
        ar = self.arabic_only + (self.arabic_yk if self.persian_only == 0 else 0)
        return "fa" if fa >= ar else "ar"

    def latin_kind(self):
        if self.latin_words < 20:
            return "en"
        return "en" if self.en_stop / max(1, self.latin_words) >= 0.06 else "latin-other"

    def label(self):
        n = self.letters
        if n < 40:
            return "none"
        a = self.scripts["arabic_script"] / n
        latin = self.scripts["latin"] / n
        if a >= 0.5:
            return self.arabic_kind()
        if latin >= 0.85:
            return self.latin_kind()
        if a >= 0.15 and latin >= 0.15:
            return "mixed"
        top = self.scripts.most_common(1)[0][0]
        return {"latin": self.latin_kind(), "arabic_script": self.arabic_kind()}.get(top, top)

    def shares(self):
        n = self.letters
        if not n:
            return {}
        out = OrderedDict()
        a = self.scripts["arabic_script"]
        if a:
            out["Persian/Arabic script" if self.arabic_kind() == "fa" else "Arabic script"] = a / n
        if self.scripts["latin"]:
            out["Latin" + (" (English)" if self.latin_kind() == "en" else "")] = self.scripts["latin"] / n
        rest = n - a - self.scripts["latin"]
        if rest:
            out["other scripts"] = rest / n
        return out


def fmt_shares(sh):
    return ", ".join("%s %d%%" % (k, round(v * 100)) for k, v in sh.items()) or "no letters found"


# ---------------------------------------------------------------- reading
class Reader:
    def __init__(self, limits):
        self.lim = limits
        self.files_read = 0
        self.bytes_read = 0
        self.capped = []
        self._cache = {}

    def can_read(self):
        if self.files_read >= self.lim["max_files_read"]:
            self._cap("max_files_read")
            return False
        if self.bytes_read >= self.lim["max_total_bytes"]:
            self._cap("max_total_bytes")
            return False
        return True

    def _cap(self, what):
        if what not in self.capped:
            self.capped.append(what)

    def raw(self, path, limit):
        if not self.can_read():
            return None
        limit = min(limit, self.lim["max_total_bytes"] - self.bytes_read)
        try:
            with open(path, "rb") as f:
                data = f.read(limit)
        except OSError:
            return None
        self.files_read += 1
        self.bytes_read += len(data)
        return data

    def text(self, path):
        """Decoded text of the first bytes of a file; each file is read at most once."""
        if path in self._cache:
            return self._cache[path]
        data = self.raw(path, self.lim["max_file_bytes"])
        txt = None if data is None or b"\x00" in data[:8192] else data.decode("utf-8", errors="replace").lstrip("\ufeff")
        if data is not None:
            self._cache[path] = txt
        return txt


TAG_RE = P(r"<[^>]+>")
SCRIPT_STYLE_RE = P(r"<(script|style)\b.*?</\1>", re.S | re.I)


def html_text(s):
    s = SCRIPT_STYLE_RE.sub(" ", s)
    return html.unescape(TAG_RE.sub(" ", s))


def json_strings(s, full):
    """String values of a JSON or JSONL document (keys dropped)."""
    out = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    if full:
        try:
            walk(json.loads(s))
            return out
        except ValueError:
            pass
        ok = False
        for line in s.splitlines():
            line = line.strip()
            if line:
                try:
                    walk(json.loads(line))
                    ok = True
                except ValueError:
                    pass
        if ok:
            return out
    for m in re.finditer(r'"((?:[^"\\]|\\.){2,})"\s*(?![:\s]*:)', s):
        try:
            out.append(json.loads('"' + m.group(1) + '"'))
        except ValueError:
            pass
    return out


def docx_text(path, reader):
    if not reader.can_read():
        return None
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > reader.lim["max_docx_xml_bytes"] * 8:
                return None
            with z.open(info) as f:
                data = f.read(reader.lim["max_docx_xml_bytes"])
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError):
        return None
    reader.files_read += 1
    reader.bytes_read += len(data)
    xml = data.decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return html.unescape(TAG_RE.sub("", xml))


PDF_STR = P(rb"\(((?:\\.|[^\\)]){1,400})\)\s*(?:Tj|')")
PDF_ARR = P(rb"\[((?:[^\]]){1,4000})\]\s*TJ")
PDF_INNER = P(rb"\(((?:\\.|[^\\)]){1,400})\)")


def _pdf_unescape(b):
    b = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8) & 255]), b)
    b = re.sub(rb"\\(.)", lambda m: {b"n": b"\n", b"t": b"\t", b"r": b"\r"}.get(m.group(1), m.group(1)), b)
    return b.decode("latin-1")


def pdf_info(path, reader):
    """Best-effort text from a PDF without a PDF library. Returns (text, pages, lang)."""
    data = reader.raw(path, reader.lim["max_pdf_bytes"])
    if data is None:
        return "", 0, None
    pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", data))
    m = re.search(rb"/Lang\s*\(([A-Za-z-]{2,12})\)", data) or re.search(rb"<dc:language>.*?<rdf:li>([A-Za-z-]{2,12})<", data, re.S)
    lang = m.group(1).decode("ascii", "ignore").lower() if m else None
    parts, n = [], 0
    for sm in re.finditer(rb"stream\r?\n", data):
        end = data.find(b"endstream", sm.end())
        if end < 0 or n > 400:
            break
        n += 1
        raw = data[sm.end():end]
        try:
            dec = zlib.decompressobj().decompress(raw, 2000000)
        except zlib.error:
            dec = raw
        found = [_pdf_unescape(x) for x in PDF_STR.findall(dec)]
        for arr in PDF_ARR.findall(dec):
            found.append("".join(_pdf_unescape(x) for x in PDF_INNER.findall(arr)))
        txt = " ".join(found)
        if txt:
            printable = sum(1 for c in txt if c.isascii() and (c.isalnum() or c in " .,;:'-()?!\n"))
            if printable / len(txt) >= 0.85:   # skip glyph-id strings (CID fonts) that decode to noise
                parts.append(txt)
        if sum(len(p) for p in parts) > reader.lim["sample_chars"]:
            break
    return " ".join(parts), pages, lang


# ---------------------------------------------------------------- walk and classify
def rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def is_test_path(r):
    parts = r.lower().split("/")
    name = parts[-1]
    return any(p in ("test", "tests", "testing", "__tests__", "spec") for p in parts[:-1]) or \
        name.startswith("test_") or re.search(r"(_test|\.test|\.spec)\.\w+$", name) is not None


def skip_dir(parent, d):
    if d in SKIP_DIRS or d.endswith(".egg-info"):
        return True
    if d.startswith(".") and d not in HIDDEN_OK:
        return True
    return os.path.exists(os.path.join(parent, d, "pyvenv.cfg"))


def walk(root, lim, stats):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not skip_dir(dirpath, d))
        for f in sorted(filenames):
            stats["seen"] += 1
            if stats["seen"] > lim["max_entries"]:
                stats["capped"] = True
                return
            p = os.path.join(dirpath, f)
            if os.path.islink(p):
                continue
            yield p


def classify(r):
    """'corpus', 'query', 'locale', 'code', 'ui', 'doc' (project docs), 'skip'."""
    parts = r.lower().split("/")
    name = parts[-1]
    stem, ext = os.path.splitext(name)
    dirs = parts[:-1]
    if ext in BINARY_EXT:
        return "skip"
    if name.startswith(".env") and not re.search(r"example|sample|template", name):
        return "skip"   # may hold secrets; never read
    if any(d in LOCALE_DIRS for d in dirs) and ext in (".json", ".yaml", ".yml", ".po", ".properties", ".js", ".ts"):
        return "locale"
    if CONFIG_NAMES.match(name):
        return "code"
    if any(d in QUERY_DIRS for d in dirs) and ext in (".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml", ".md"):
        return "query"
    if ext in DOC_EXT:
        if NOT_CORPUS_NAME.match(stem):
            return "doc" if ext in (".md", ".txt", ".html", ".htm") else "code"
        if QUERY_NAME.search(stem) and ext in (".json", ".jsonl", ".txt", ".md", ".csv"):
            return "query"
        if any(d in NOT_CORPUS_DIRS for d in dirs):
            return "doc" if ext in (".md", ".txt") else "skip"
        if ext in (".pdf", ".docx"):
            return "corpus"
        if any(d in CORPUS_DIRS for d in dirs):
            return "corpus"
        if dirs and dirs[0] == "docs" and ext in (".html", ".htm", ".txt"):
            return "corpus"
        if ext in (".html", ".htm"):
            return "ui"
        if ext in (".json", ".jsonl"):
            return "code"
        return "doc"
    if ext == ".csv" and QUERY_NAME.search(stem):
        return "query"
    if ext in UI_EXT:
        return "ui"
    if ext in CODE_EXT:
        return "code"
    return "skip"


def loc(r, i):
    return "%s:%d" % (r, i)


# ---------------------------------------------------------------- scanning
def scan(root, limits=None):
    lim = dict(LIMITS, **(limits or {}))
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise SystemExit("not a directory: " + root)
    reader = Reader(lim)
    wstats = {"seen": 0, "capped": False}
    files = {"corpus": [], "query": [], "locale": [], "code": [], "ui": [], "doc": []}
    for p in walk(root, lim, wstats):
        r = rel(root, p)
        kind = classify(r)
        if kind == "skip":
            continue
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        files[kind].append((p, r, size))

    # A folder of 10+ plain .md/.txt files outside code is probably a corpus even without a telling name.
    by_dir = Counter(os.path.dirname(r) for _, r, _ in files["doc"] if r.lower().endswith((".md", ".txt")))
    keep = []
    for item in files["doc"]:
        d = os.path.dirname(item[1])
        if by_dir[d] >= 10 and d and not d.lower().startswith(("docs", ".github")):
            files["corpus"].append(item)
        else:
            keep.append(item)
    files["doc"] = keep

    out = OrderedDict()
    out["tool"] = "choose-rag/scan_project.py"
    out["version"] = VERSION
    out["root"] = root
    corpus = scan_corpus(files["corpus"], reader, lim)
    stack, resources, hints = scan_code(files["code"] + files["ui"], reader)
    queries = scan_queries(files, reader, lim)
    hints.update(scan_hints(files, reader, corpus))
    out["corpus"] = corpus
    out["queries"] = queries
    out["stack"] = stack
    out["resources"] = resources
    slots = infer_slots(corpus, queries, stack, resources, hints)
    out["slots"] = slots
    out["answers"] = OrderedDict((q, s["value"]) for q, s in slots.items() if s["value"])
    out["ask"] = [q for q, s in slots.items() if CONF_RANK[s["confidence"]] <= 1 and s.get("applies", True)]
    out["confirm"] = [q for q, s in slots.items() if CONF_RANK[s["confidence"]] >= 2]
    out["out_of_scope"] = scope_notes(corpus, queries, stack)
    out["scanned"] = {"files_seen": wstats["seen"], "files_read": reader.files_read,
                      "bytes_read": reader.bytes_read, "caps_hit": reader.capped + (["max_entries"] if wstats["capped"] else []),
                      "limits": lim, "executed_code": False, "network": False}
    return out


def scan_corpus(items, reader, lim):
    c = OrderedDict()
    c["files"] = len(items)
    c["bytes"] = sum(s for _, _, s in items)
    types = Counter(DOC_EXT.get(os.path.splitext(r)[1].lower(), "other") for _, r, _ in items)
    c["types"] = dict(types.most_common())
    folders = Counter()
    for _, r, _ in items:
        parts = r.split("/")
        folders["/".join(parts[:2]) + "/" if len(parts) > 2 else (parts[0] + "/" if len(parts) > 1 else "./")] += 1
    c["folders"] = dict(folders.most_common(8))
    total = TextStats()
    fa_total = TextStats()      # Persian files only, for the q8 spelling check
    labels = Counter()
    ratio = {}            # type -> [chars, bytes] from samples
    pdf_pages = pdf_no_text = 0
    pdf_lang = Counter()
    sample = items
    if len(items) > lim["max_corpus_samples"]:
        step = len(items) / float(lim["max_corpus_samples"])
        sample = [items[int(i * step)] for i in range(lim["max_corpus_samples"])]
    sampled = 0
    per_file = []
    for p, r, size in sample:
        t = DOC_EXT.get(os.path.splitext(r)[1].lower())
        text, read_bytes = None, size
        if t == "pdf":
            text, pages, lang = pdf_info(p, reader)
            pdf_pages += pages
            if lang:
                pdf_lang[lang] += 1
            if len(text.strip()) < 200:
                pdf_no_text += 1
                if lang:
                    labels["fa" if lang.startswith("fa") else "en" if lang.startswith("en") else lang] += 1
                continue
            read_bytes = min(size, lim["max_pdf_bytes"])
        elif t == "docx":
            text = docx_text(p, reader)
        else:
            raw = reader.text(p)
            if raw is None:
                continue
            read_bytes = min(size, len(raw.encode("utf-8", "replace")))
            if t == "html":
                text = html_text(raw)
            elif t == "json":
                text = "\n".join(json_strings(raw, size <= lim["max_file_bytes"]))
            else:
                text = raw
        if not text:
            continue
        sampled += 1
        text = text[:lim["sample_chars"]]
        st = TextStats()
        st.add(text)
        total.merge(st)
        lab = st.label()
        if lab == "fa":
            fa_total.merge(st)
        labels[lab] += 1
        per_file.append((r, lab))
        if t in ("txt", "md", "html", "json") and read_bytes:
            acc = ratio.setdefault(t, [0, 0])
            full = len(text) if len(text) < lim["sample_chars"] else None
            if full is not None:
                acc[0] += full
                acc[1] += read_bytes
    # rough characters
    chars, how = 0, []
    for t in types:
        tb = sum(s for _, r, s in items if DOC_EXT.get(os.path.splitext(r)[1].lower()) == t)
        if t in ratio and ratio[t][1]:
            chars += int(tb * ratio[t][0] / ratio[t][1])
        elif t in ("txt", "md"):
            chars += tb
        elif t == "pdf" and pdf_pages:
            n_pdf = types["pdf"]
            sampled_pdf = sum(1 for _, r, _ in sample if r.lower().endswith(".pdf"))
            est_pages = pdf_pages * n_pdf / max(1, sampled_pdf)
            chars += int(est_pages * 2500)
            how.append("PDF from page count x 2,500 characters (rough)")
        elif t == "docx":
            chars += tb * 2
            how.append("docx from file size (rough)")
    c["chars_estimate"] = chars
    c["chars_method"] = "; ".join(how) or "text sample ratio x total size"
    c["sampled_files"] = sampled
    c["file_languages"] = dict(labels.most_common())
    c["script_shares"] = {k: round(v, 3) for k, v in total.shares().items()}
    c["pdf"] = {"files": types.get("pdf", 0), "pages_in_sample": pdf_pages,
                "without_readable_text": pdf_no_text, "declared_lang": dict(pdf_lang)}
    c["_stats"] = total
    c["_fa_stats"] = fa_total
    c["_per_file"] = per_file[:50]
    return c


QKEY = P(r"^(question|questions|query|queries|q|prompt|input|user_query|user_question|ask)(_[a-z]{2,8})?$", re.I)
QLINE = P(r"^\s*-?\s*['\"]?(?:question|query|q|prompt|input)(?:_[a-z]{2,8})?['\"]?\s*[:=]\s*['\"]?(.{8,400}?)['\"]?,?\s*$", re.I)


def question_texts(raw, ext, full):
    """The questions themselves: values under question-like keys, else lines ending in '?'."""
    out = []

    def walk(v, key=None):
        if isinstance(v, str):
            if key and QKEY.match(key) and 8 <= len(v) < 400:
                out.append(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, str(k).lower())
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
    if ext in (".json", ".jsonl") and full:
        try:
            walk(json.loads(raw))
        except ValueError:
            for line in raw.splitlines():
                try:
                    walk(json.loads(line))
                except ValueError:
                    pass
    if not out:
        for line in raw.splitlines():
            m = QLINE.match(line)
            if m:
                out.append(m.group(1))
            elif line.strip().endswith(("?", "\u061f")) and 8 <= len(line.strip()) < 400:
                out.append(line.strip().lstrip("-*0123456789. "))
    return out


def scan_queries(files, reader, lim):
    q = OrderedDict()
    srcs = OrderedDict([("questions", TextStats()), ("ui", TextStats()), ("prompts", TextStats())])
    where = {k: [] for k in srcs}
    questions = []
    for p, r, size in files["query"]:
        raw = reader.text(p)
        if raw is None:
            continue
        ext = os.path.splitext(r)[1].lower()
        texts = json_strings(raw, size <= lim["max_file_bytes"]) if ext in (".json", ".jsonl") else raw.splitlines()
        texts = [x for x in texts if len(x) >= 8]
        for x in texts:
            srcs["questions"].add(x)
        questions += question_texts(raw, ext, size <= lim["max_file_bytes"])[:500]
        where["questions"].append(r)
    qlab = Counter()
    fa_q = TextStats()
    for x in questions:
        st = TextStats()
        st.add(x)
        if st.letters >= 8:
            n = st.letters
            a = st.scripts["arabic_script"] / n
            lab = ("fa" if a >= 0.5 and st.arabic_kind() == "fa" else "ar" if a >= 0.5 else
                   "en" if st.scripts["latin"] / n >= 0.85 else "mixed")
            qlab[lab] += 1
            if lab == "fa":
                fa_q.merge(st)
    for p, r, size in files["locale"]:
        raw = reader.text(p)
        if raw is None:
            continue
        srcs["ui"].add("\n".join(json_strings(raw, True)) if r.endswith(".json") else raw)
        where["ui"].append(r)
    for p, r, size in files["ui"]:
        raw = reader.text(p)
        if raw is None:
            continue
        t = html_text(raw) if r.lower().endswith((".html", ".htm")) else " ".join(
            m.group(2) for m in STRING_LIT.finditer(raw)) + " " + " ".join(re.findall(r">([^<>{}]{3,200})<", raw))
        st = TextStats()
        st.add(t)
        if st.letters >= 20:
            srcs["ui"].merge(st)
            where["ui"].append(r)
    for p, r, size in files["code"]:
        if not r.lower().endswith((".py", ".js", ".ts", ".ipynb", ".yaml", ".yml", ".toml", ".txt", ".md")):
            continue
        if is_test_path(r) or CONFIG_NAMES.match(r.split("/")[-1]):
            continue
        raw = reader.text(p)
        if raw is None:
            continue
        hit = False
        for line in raw.splitlines():
            if not PROMPTISH.search(line) and not re.search(r"[\u0600-\u06ff]", line):
                continue
            for m in STRING_LIT.finditer(line):
                s = m.group(2)
                if re.search(r"[A-Za-z\u0600-\u06ff]{3}", s) and not re.match(r"^[\w./:-]+$", s):
                    srcs["prompts"].add(s)
                    hit = True
        if hit:
            where["prompts"].append(r)
    for k, st in srcs.items():
        q[k] = {"files": len(where[k]), "examples": where[k][:4], "letters": st.letters,
                "language": st.label() if st.letters >= 40 else "unknown",
                "script_shares": {a: round(b, 3) for a, b in st.shares().items()}}
    q["questions"]["per_question"] = dict(qlab.most_common())
    q["_stats"] = srcs
    q["_fa_questions"] = fa_q
    q["_questions"] = questions
    return q


def add_ev(d, key, label, where, cap=4):
    e = d.setdefault(key, OrderedDict()).setdefault(label, {"count": 0, "where": []})
    e["count"] += 1
    if len(e["where"]) < cap and where not in e["where"]:
        e["where"].append(where)


def num(s):
    return int(s.replace("_", "").replace(",", ""))


def scan_code(items, reader):
    stack, res, hints = OrderedDict(), OrderedDict(), {"timeouts": []}
    settings = {"chunk_size": [], "chunk_overlap": [], "top_k": [], "context_budget": []}
    for p, r, size in items:
        raw = reader.text(p)
        if raw is None:
            continue
        test = is_test_path(r)
        is_config = bool(CONFIG_NAMES.match(r.split("/")[-1])) or r.lower().endswith((".yaml", ".yml", ".toml", ".cfg", ".sh"))
        file_tokens = bool(TOKEN_SPLITTER.search(raw))
        for i, line in enumerate(raw.splitlines(), 1):
            if len(line) > 2000:
                line = line[:2000]
            for cat, pats in STACK_PATTERNS.items():
                for label, rx in pats:
                    if rx.search(line):
                        add_ev(stack, cat, label, loc(r, i))
            for cat, pats in RESOURCE_PATTERNS.items():
                if cat.startswith("latency") and test:
                    continue
                for label, rx in pats:
                    if cat == "gpu_infra" and not is_config and label != "vLLM (GPU server)":
                        continue
                    if rx.search(line):
                        add_ev(res, cat, label, loc(r, i))
            for m in TIMEOUT_RE.finditer(line):
                hints["timeouts"].append((float(m.group(1)), loc(r, i)))
            for m in CHUNK_RE.finditer(line):
                name, v = m.group(1), num(m.group(2))
                unit = "tokens" if ("token" in name.lower() or (file_tokens and "char" not in name.lower())) else "chars"
                if 50 <= v <= 100000:
                    settings["chunk_size"].append({"name": name, "value": v, "unit": unit, "where": loc(r, i), "test": test})
            for m in OVERLAP_RE.finditer(line):
                settings["chunk_overlap"].append({"name": m.group(1), "value": num(m.group(2)), "where": loc(r, i), "test": test})
            for m in TOPK_RE.finditer(line):
                name, v = m.group(1), int(m.group(2))
                if name.lower() in ("k", "limit") and not TOPK_CONTEXT.search(line):
                    continue
                if 1 <= v <= 500:
                    settings["top_k"].append({"name": name, "value": v, "where": loc(r, i), "test": test})
            for m in BUDGET_RE.finditer(line):
                name, v = m.group(1), num(m.group(2))
                if "max_tokens" == name.lower() or v < 200:
                    continue
                unit = "tokens" if "token" in name.lower() else ("chars" if "char" in name.lower() or v >= 2000 else "unknown")
                settings["context_budget"].append({"name": name, "value": v, "unit": unit, "where": loc(r, i), "test": test})
    for k, v in settings.items():
        stack[k] = v
    # elasticsearch/opensearch count as BM25 only when no other BM25 evidence exists
    return stack, res, hints


def scan_hints(files, reader, corpus):
    """Keyword hints for q1 (question type) and q7 (public vs private) from docs, prompts and paths."""
    h = {"q1_single": [], "q1_cross": [], "q1_within": [], "q7_private": [], "q7_public": []}
    texts = []
    for p, r, size in files["doc"]:          # project docs (README etc.); question files are read per question
        raw = reader.text(p)
        if raw is not None:
            texts.append((r, raw))
    for r, raw in texts:
        readme = r.split("/")[-1].lower().startswith("readme")
        for i, line in enumerate(raw.splitlines(), 1):
            for key, rx in (("q1_single", Q1_SINGLE), ("q1_cross", Q1_CROSS), ("q7_public", Q7_PUBLIC),
                            ("q7_private", Q7_PRIVATE if readme else None)):
                if rx and rx.search(line) and len(h[key]) < 50:
                    h[key].append(loc(r, i))
    for _, r, _ in files["corpus"] + files["query"]:
        if Q7_PRIVATE.search(r) and len(h["q7_private"]) < 50:
            h["q7_private"].append(r)
        if Q7_PUBLIC.search(r) and len(h["q7_public"]) < 50:
            h["q7_public"].append(r)
        if Q1_SINGLE.search(r) and len(h["q1_single"]) < 50:
            h["q1_single"].append(r)
    return h


# ---------------------------------------------------------------- slot inference
def slot(value, conf, evidence, why, applies=True):
    return OrderedDict([("value", value), ("confidence", conf if value else "unknown"),
                        ("evidence", evidence[:6]), ("why", why), ("applies", applies)])


def first_where(cat_dict, cap=3):
    out = []
    for label, e in cat_dict.items():
        out.append("%s (%s)" % (label, ", ".join(e["where"][:cap]) + (" +%d more" % (e["count"] - len(e["where"][:cap]))
                                                                      if e["count"] > cap else "")))
    return out


def corpus_lang(corpus):
    """('fa'|'en'|'both'|'ar'|'other'|None, confidence, evidence)"""
    fl = Counter(corpus.get("file_languages", {}))
    fl.pop("none", None)
    n = sum(fl.values())
    if not n:
        return None, "unknown", []
    sh = corpus["script_shares"]
    ev = ["%d corpus files sampled: %s" % (n, ", ".join("%s %d" % kv for kv in fl.most_common())),
          "script sample: " + fmt_shares(sh)]
    fa, en = fl["fa"], fl["en"]
    conf = "high" if n >= 5 else "medium" if n >= 2 else "low"
    if fa and en and min(fa, en) >= 0.15 * n:
        return "both", ("medium" if n >= 4 else "low"), ev
    top, cnt = fl.most_common(1)[0]
    if top == "mixed":
        stats = corpus["_stats"]
        a = stats.scripts["arabic_script"] / max(1, stats.letters)
        return ("fa" if a >= 0.4 and stats.arabic_kind() == "fa" else "both"), "low", ev
    if cnt < 0.7 * n:
        conf = "low"
    return top, conf, ev


def query_lang(queries):
    pq = Counter(queries["questions"].get("per_question", {}))
    pq.pop("mixed", None)
    n = sum(pq.values())
    if n >= 5:
        ev = ["%d test questions: %s (%s)" % (n, ", ".join("%s %d" % kv for kv in pq.most_common()),
                                               ", ".join(queries["questions"]["examples"][:3]))]
        fa, en = pq["fa"], pq["en"]
        if fa and en and min(fa, en) >= 0.15 * n:
            return "both", "medium", "questions", ev
        top, cnt = pq.most_common(1)[0]
        return top, ("medium" if cnt >= 0.8 * n else "low"), "questions", ev
    for src, conf in (("questions", "medium"), ("ui", "low"), ("prompts", "low")):
        s = queries[src]
        if s["language"] not in ("unknown", "none"):
            return s["language"], conf, src, ["%s: %s in %s" % (src, fmt_shares(s["script_shares"]), ", ".join(s["examples"][:3]))]
    return None, "unknown", None, []


def infer_slots(corpus, queries, stack, res, hints):
    S = OrderedDict()

    # q1 question type
    qs = queries.get("_questions", [])
    cross_q = [x for x in qs if Q1_CROSS.search(x)]
    ev1 = []
    if len(qs) >= 5:
        share = len(cross_q) / len(qs)
        ev1.append("%d of %d test questions look cross-document (compare / difference / across)" % (len(cross_q), len(qs)))
        v = "t3" if share >= 0.6 else "mix" if share >= 0.2 else None
        if v:
            S["q1"] = slot(v, "medium" if len(qs) >= 15 else "low", ev1, "from the wording of the project's test questions")
    if "q1" not in S:
        single, cross = hints["q1_single"], hints["q1_cross"]
        if single and not cross:
            S["q1"] = slot("t1", "low", ["FAQ/support wording: " + ", ".join(single[:4])], "FAQ or support bot: mostly single facts")
        elif cross and not single:
            S["q1"] = slot("t3", "low", ["comparison/analysis wording: " + ", ".join(cross[:4])], "comparison or analysis across documents")
        elif single and cross:
            S["q1"] = slot("mix", "low", ["FAQ wording: " + ", ".join(single[:3]), "comparison wording: " + ", ".join(cross[:3])],
                           "both FAQ and comparison wording")
        elif len(qs) >= 5:
            S["q1"] = slot(None, "unknown", ev1, "test questions found, but none look cross-document: single facts or facts within one document")
        else:
            S["q1"] = slot(None, "unknown", [], "no FAQ, comparison or test-question evidence")

    # q2 language
    cl, cconf, cev = corpus_lang(corpus)
    ql, qconf, qsrc, qev = query_lang(queries)
    ev2 = cev + qev
    if cl in ("ar", "latin-other", "cyrillic", "cjk", "other", "hebrew", "devanagari", "greek", "mixed"):
        S["q2"] = slot(None, "unknown", ev2, "corpus language (%s) is outside the study's Persian/English" % cl)
    elif cl in ("fa", "en"):
        if ql in ("fa", "en") and ql != cl and qsrc in ("questions", "ui"):
            S["q2"] = slot("cross", "medium" if qsrc == "questions" else "low", ev2,
                           "%s documents, %s questions/UI" % (cl, ql))
        else:
            conf = cconf
            if ql == cl and qsrc == "questions" and conf == "medium":
                conf = "high"
            elif ql is None and conf == "high":
                conf = "medium"
            why = "%s documents; %s" % (cl, "query side agrees" if ql == cl else
                                         "query side unknown" if ql is None else
                                         "prompts are in %s, which is common and not taken as the question language" % ql)
            S["q2"] = slot(cl, conf, ev2, why)
    elif cl == "both":
        S["q2"] = slot("both", cconf, ev2, "Persian and English document collections; assumed each asked in its own language")
    elif ql in ("fa", "en", "both"):
        S["q2"] = slot(ql, "low", ev2, "no corpus text found; from the %s only" % qsrc)
    else:
        S["q2"] = slot(None, "unknown", ev2, "no corpus text or query-side text found")

    persian = S["q2"]["value"] in ("fa", "both", "cross") or cl in ("fa", "both") or ql in ("fa", "both")

    # q3 budget
    S["q3"] = infer_budget(stack, S["q2"]["value"])

    # q4 latency
    fast = res.get("latency_fast", {})
    stream = res.get("latency_stream", {})
    batch = res.get("latency_batch", {})
    tos = sorted(t for t in hints.get("timeouts", []) if t[0] > 0)
    if fast and batch:
        S["q4"] = slot(None, "unknown", first_where(fast)[:1] + first_where(batch)[:1],
                       "both real-time and batch/offline hints")
    elif fast:
        S["q4"] = slot("fast", "medium" if len(fast) >= 2 else "low", first_where(fast), "search-as-you-type or real-time hints")
    elif stream:
        S["q4"] = slot("few", "low", first_where(stream), "streaming chat answers usually tolerate a few seconds")
    elif batch:
        S["q4"] = slot("slow", "low", first_where(batch), "batch or offline processing")
    else:
        S["q4"] = slot(None, "unknown", ["timeouts: " + ", ".join("%gs at %s" % t for t in tos[:3])] if tos else [],
                       "no latency hints (timeouts are upper bounds, not targets)")

    # q5 GPU
    infra, code_gpu, cpu = res.get("gpu_infra", {}), res.get("gpu_code", {}), res.get("cpu_only", {})
    local = res.get("local_models", {})
    prov = res.get("llm_provider", {})
    hosted = [k for k in prov if k not in ("ollama", "vllm", "local transformers", "llama.cpp")]
    reranker_local = any(k in stack.get("reranker", {}) for k in ("bge-reranker", "cross-encoder"))
    if infra:
        S["q5"] = slot("yes", "high" if len(infra) >= 2 or code_gpu else "medium", first_where(infra) + first_where(code_gpu),
                       "CUDA in requirements, Docker or a GPU inference server")
    elif code_gpu and (not cpu or sum(e["count"] for e in code_gpu.values()) > sum(e["count"] for e in cpu.values())):
        S["q5"] = slot("yes", "low", first_where(code_gpu),
                       "code uses CUDA (often with a CPU fallback); confirm the deployment has a GPU")
    elif cpu:
        S["q5"] = slot("no", "medium" if not code_gpu else "low", first_where(cpu), "CPU-only packages or device")
    elif hosted and not local and not reranker_local:
        S["q5"] = slot("no", "low", ["hosted LLM API only: " + ", ".join(hosted)],
                       "hosted APIs only and no local model libraries")
    else:
        S["q5"] = slot(None, "unknown", first_where(local)[:2], "no GPU or CPU-only hints")

    # q6 extra LLM calls
    agent, rew, sel = stack.get("agent", {}), stack.get("query_rewriting", {}), stack.get("llm_selection", {})
    if agent:
        S["q6"] = slot("many", "low", first_where(agent), "the project already runs an agent loop (several LLM calls)")
    elif rew or sel:
        S["q6"] = slot("one", "low", first_where(rew) + first_where(sel), "the project already makes an extra LLM call per question")
    else:
        S["q6"] = slot(None, "unknown", [], "nothing shows whether extra LLM calls are acceptable")

    # q7 public vs private
    pv, pu = hints["q7_private"], hints["q7_public"]
    if pv and not pu:
        S["q7"] = slot("private", "low", ["private-looking names: " + ", ".join(pv[:4])], "internal, policy, contract or report documents")
    elif pu and not pv:
        S["q7"] = slot("public", "low", ["public-text names: " + ", ".join(pu[:4])], "well-known public texts")
    else:
        S["q7"] = slot(None, "unknown", (["private: " + ", ".join(pv[:2]), "public: " + ", ".join(pu[:2])] if pv else []),
                       "cannot tell whether the model already knows these documents")

    # q8 Persian spelling
    st = corpus.get("_fa_stats")
    q8_src = "corpus"
    if (not st or st.scripts["arabic_script"] < 200) and queries["_fa_questions"].scripts["arabic_script"] >= 200:
        st, q8_src = queries["_fa_questions"], "test questions"
    if not persian:
        S["q8"] = slot(None, "unknown", [], "not asked: no Persian found", applies=False)
    elif not st or st.scripts["arabic_script"] < 200:
        S["q8"] = slot(None, "unknown", [], "too little Persian corpus text to check spelling")
    else:
        yk = st.persian_yk + st.arabic_yk
        ar_share = st.arabic_yk / max(1, yk)
        dig = st.digits
        mixed_digits = sum(1 for k in ("persian", "arabic_indic") if dig[k] > 5) + (1 if dig["ascii"] > 20 and (dig["persian"] > 5 or dig["arabic_indic"] > 5) else 0)
        mi_total = st.mi_spaced + st.mi_joined
        spaced = st.mi_spaced / mi_total if mi_total >= 5 else 0
        ev = ["Arabic \u064a/\u0643 vs Persian \u06cc/\u06a9: %d vs %d (%.1f%% Arabic forms)" % (st.arabic_yk, st.persian_yk, 100 * ar_share),
              "digits: ascii %d, Persian %d, Arabic-Indic %d" % (dig["ascii"], dig["persian"], dig["arabic_indic"]),
              "non-joiners: %d; 'mi' prefix spaced %d vs joined %d" % (st.zwnj, st.mi_spaced, st.mi_joined)]
        signs = (ar_share >= 0.02) + (mixed_digits >= 2) + (spaced >= 0.3)
        cap = "low" if q8_src != "corpus" else None
        if signs:
            S["q8"] = slot("mixed", cap or ("high" if ar_share >= 0.1 or signs >= 2 else "medium"), ev,
                           "spelling varies in the %s sample" % q8_src)
        else:
            S["q8"] = slot("clean", cap or ("medium" if st.scripts["arabic_script"] >= 5000 else "low"), ev,
                           "no Arabic letter forms or mixed digits in the %s sample" % q8_src)
    return S


def chunk_chars(entry, lang):
    if entry["unit"] == "tokens":
        return int(entry["value"] * (2.1 if lang == "fa" else 3.5))
    return entry["value"]


def primary(entries):
    """Most common non-test value (first location kept), or test values if nothing else."""
    pool = [e for e in entries if not e["test"]] or entries
    if not pool:
        return None, 0
    cnt = Counter(e["value"] for e in pool)
    v, _ = cnt.most_common(1)[0]
    return next(e for e in pool if e["value"] == v), len(cnt)


def pack_estimate(stack, lang):
    """(chars, how, confidence, where list) for the current evidence budget, or None."""
    b, nb = primary(stack.get("context_budget", []))
    if b and b["unit"] in ("chars", "tokens"):
        chars = b["value"] if b["unit"] == "chars" else int(b["value"] * (2.1 if lang == "fa" else 3.5))
        return chars, "%s=%d %s" % (b["name"], b["value"], b["unit"]), ("medium" if nb == 1 else "low"), [b["where"]]
    k, nk = primary(stack.get("top_k", []))
    c, nc = primary(stack.get("chunk_size", []))
    if k and c:
        cc = chunk_chars(c, lang)
        chars = int(k["value"] * cc * 0.81)
        how = "top_k=%d x %s=%d %s x 0.81 (mean fill of a chunk)" % (k["value"], c["name"], c["value"], c["unit"])
        return chars, how, ("medium" if nk == 1 and nc == 1 else "low"), [k["where"], c["where"]]
    return None


def infer_budget(stack, lang):
    pe = pack_estimate(stack, lang)
    if not pe:
        k, _ = primary(stack.get("top_k", []))
        ev = ["top_k=%d at %s, chunk size not found" % (k["value"], k["where"])] if k else []
        return slot(None, "unknown", ev, "no context budget or top_k x chunk_size found")
    chars, how, conf, where = pe
    v = "tight" if chars <= 4500 else "moderate" if chars <= 10000 else "generous"
    return slot(v, conf, ["about %s characters: %s (%s)" % (format(chars, ","), how, ", ".join(where))],
                "current pack size; the slot asks what the reader MAY see, so confirm")


def scope_notes(corpus, queries, stack):
    n = []
    cl = corpus_lang(corpus)[0]
    if cl in ("ar", "latin-other", "cyrillic", "cjk", "other", "hebrew", "devanagari", "greek"):
        n.append("Corpus language (%s) is outside the study (Persian and English only)." % cl)
    if corpus["chars_estimate"] > 380000000 or corpus["files"] > 20000:
        n.append("Corpus is much larger than the study's 155 documents / ~38 M characters.")
    pdf = corpus["pdf"]
    if pdf["files"] and pdf["without_readable_text"]:
        n.append("%d sampled PDF(s) had no text this scanner could read (CID fonts or scans); scans/OCR are outside the study."
                 % pdf["without_readable_text"])
    for cat, msg in (("graph_or_sql", "Graph or SQL retrieval found; GraphRAG, tables and SQL are outside the study."),
                     ("chat_memory", "Multi-turn chat memory found; the study tested single questions with short answers.")):
        if stack.get(cat):
            n.append(msg + " (" + ", ".join(first_where(stack[cat], 2)) + ")")
    return n


# ---------------------------------------------------------------- changes to the user's code
TARGET = {"bd.tight": 3000, "bd.tight_ce": 3000, "bd.tight_t1": 3000, "bd.moderate": 6000, "bd.modest_t1": 6000,
          "bd.generous": 15500, "bd.raise": 15500}


def wfirst(stack, cat, prefer=None):
    """First non-test location for a stack category (optionally a given label)."""
    d = stack.get(cat, {})
    locs = []
    for label, e in d.items():
        if prefer and label not in prefer:
            continue
        locs += e["where"]
    good = [l for l in locs if not is_test_path(l.rsplit(":", 1)[0])]
    return (good or locs or [None])[0]


def plan_changes(scan_out, state, chooser, evidence, lang=None):
    stack = scan_out["stack"]
    lang = lang or scan_out["answers"].get("q2")
    opts = chooser["options"]
    elev = {e["id"]: e["confidence"] for e in evidence}
    out = []

    def add(where, text, oid, status="change"):
        o = opts[oid]
        out.append(OrderedDict([("status", status), ("where", where or "(no matching file found)"), ("change", text),
                                ("option", oid), ("confidence", choose.conf_text(o, chooser, "en") or "-"),
                                ("evidence", ["%s [%s]" % (e, elev.get(e, "?")) for e in o.get("evidence", [])])]))

    vdb = list(stack.get("vector_db", {}))
    bm = [k for k in stack.get("bm25", {}) if k not in ("elasticsearch", "opensearch") or not vdb or k in vdb]
    dense = bool(vdb or stack.get("embedding"))
    has_any = dense or bm
    retr = wfirst(stack, "retrieval_call") or wfirst(stack, "vector_db") or wfirst(stack, "bm25")
    names = {"chroma": "Chroma", "faiss": "FAISS", "qdrant": "Qdrant", "weaviate": "Weaviate", "milvus": "Milvus",
             "lancedb": "LanceDB", "elasticsearch": "Elasticsearch", "opensearch": "OpenSearch", "pinecone": "Pinecone"}
    dbname = names.get((vdb or [""])[0], (vdb or ["vector"])[0])

    # channels
    ch = state["channels"]
    if dense and not bm:
        add(retr, "Add a BM25 channel (e.g. rank_bm25 or bm25s) next to your %s search and merge both ranked lists "
                  "with reciprocal rank fusion (RRF)." % dbname, ch)
    elif bm and not dense:
        add(wfirst(stack, "bm25"), "Add a dense channel (bge-m3) next to your BM25 and fuse with RRF; BM25 alone is "
                                   "what the study found weakest.", "av.bm25_alone_xl" if ch == "ch.cross" else "av.bm25_alone")
    elif bm and dense and not stack.get("fusion"):
        add(retr, "You have both BM25 and dense retrieval but no reciprocal rank fusion was found; merge the two "
                  "lists with RRF.", ch)
    elif bm and dense:
        add(wfirst(stack, "fusion"), "BM25 + dense with fusion: matches the recommendation.", ch, "keep")

    # chunking
    c, _ = primary(stack.get("chunk_size", []))
    if c:
        cc = chunk_chars(c, lang)
        unit = "" if c["unit"] == "chars" else " (%s tokens, about %s characters)" % (c["value"], format(cc, ","))
        if cc >= 1500:
            add(c["where"], "Your %s=%d%s: consider about 1,000 characters with 200 overlap." % (c["name"], c["value"], unit),
                "av.chunks2000")
        elif cc <= 700:
            add(c["where"], "Your %s=%d%s: consider about 1,000 characters with 200 overlap." % (c["name"], c["value"], unit),
                state["chunking"])
        else:
            add(c["where"], "Your %s=%d%s is close to the recommended ~1,000 characters." % (c["name"], c["value"], unit),
                state["chunking"], "keep")
    elif has_any:
        add(None, "No chunk size setting found; the recommendation is about 1,000 characters with 200 overlap.",
            state["chunking"], "check")

    # reranking or selection
    rr = state["reranking_or_selection"]
    rer = stack.get("reranker", {})
    sel = stack.get("llm_selection", {})
    if not has_any:
        pass                    # no retrieval code found: nothing to change, build from the recommendation
    elif rr in ("rr.ce", "rr.ce_xl", "rr.ce_fast"):
        if rer:
            add(wfirst(stack, "reranker"), "Your reranker (%s) matches the recommendation." % ", ".join(rer), rr, "keep")
        else:
            add(retr, "Add a cross-encoder reranker (e.g. bge-reranker-v2-m3) after your %s retrieval: re-score the "
                      "top ~50 fused candidates, then fill the pack in that order%s." %
                (dbname if dense else "BM25", " (only if it fits your latency budget)" if rr == "rr.ce_fast" else ""), rr)
        if sel and rer:
            add(wfirst(stack, "llm_selection"), "You stack an LLM selection step on top of a cross-encoder.", "av.sel_over_ce")
    elif rr in ("rr.sel", "rr.sel_6000"):
        if sel:
            add(wfirst(stack, "llm_selection"), "Your LLM selection step matches the recommendation.", rr, "keep")
        else:
            add(retr, "Add one LLM call after retrieval that picks the passages for the answer from the fused "
                      "candidates (no GPU needed).", rr)
        if rer:
            add(wfirst(stack, "reranker"), "Your reranker (%s) needs a GPU, which you said you do not have; the rules "
                                           "recommend the LLM selection call instead, not both." % ", ".join(rer), "av.sel_over_ce")
    elif rr in ("rr.none_t1", "rr.none_fast", "rr.none_limited", "rr.loose") and (rer or sel):
        add(wfirst(stack, "reranker") or wfirst(stack, "llm_selection"),
            "Your %s is optional here: %s." % (", ".join(list(rer) + list(sel)), opts[rr]["text"]["en"].rstrip(".")), rr, "optional")

    # budget
    target = TARGET.get(state["budget"])
    pe = pack_estimate(stack, lang)
    if target:
        k, _ = primary(stack.get("top_k", []))
        if pe:
            chars, how, _, where = pe
            if not (0.67 <= chars / float(target) <= 1.5):
                tip = (" e.g. top_k=%d with 1,000-character chunks" % max(1, round(target / 810.0))) if k else ""
                add(where[0], "Your pack is about %s characters (%s): aim for about %s characters%s." %
                    (format(chars, ","), how, format(target, ","), tip), state["budget"])
            else:
                add(where[0], "Your pack of about %s characters fits the recommended ~%s." % (format(chars, ","), format(target, ",")),
                    state["budget"], "keep")
        elif k:
            add(k["where"], "Set top_k so the pack holds about %s characters (about %d chunks of 1,000 characters)."
                % (format(target, ","), max(1, round(target / 810.0))), state["budget"], "check")

    # agent
    ag = state["agent"]
    agent = stack.get("agent", {})
    if ag.startswith("ag.none") and agent:
        add(wfirst(stack, "agent"), "Your agent loop: under your answers the rules recommend no agent. %s." % opts[ag]["text"]["en"].rstrip("."), ag)
    elif ag == "ag.nogpu_fa" and agent:
        add(wfirst(stack, "agent"), "Lexical-only agents gained nothing on Persian documents.", "av.lexical_agent_fa")
    elif not ag.startswith("ag.none") and ag != "ag.nogpu_fa" and not agent and has_any:
        add(retr, "Optional: " + opts[ag]["text"]["en"].rstrip(".") + ".", ag, "optional")

    # normalisation
    nm = state["normalisation"]
    if nm in ("nm.fa", "nm.fa_mixed") and has_any and lang in ("fa", "both", "cross"):
        if stack.get("persian_normaliser"):
            add(wfirst(stack, "persian_normaliser"), "Persian normalisation found: keep it on both documents and queries.", nm, "keep")
        else:
            add(wfirst(stack, "bm25") or retr, "Normalise Persian before BM25 indexing and on every query (Arabic "
                                               "\u064a/\u0643 to \u06cc/\u06a9, non-joiners, digits).", nm)

    # avoid
    if "av.rewrite" in state["avoid"] and stack.get("query_rewriting"):
        add(wfirst(stack, "query_rewriting"), "Your query rewriting (HyDE / multi-query) did not help in the study; "
                                              "consider removing it.", "av.rewrite")
    if "av.long_context" in state["avoid"] and stack.get("long_context"):
        add(wfirst(stack, "long_context"), "You put whole documents into the prompt; the study found retrieval better.", "av.long_context")

    # embedding scope
    emb = list(stack.get("embedding", {}))
    if emb and "ex.embedding" in state["extras"] and not ({"bge-m3", "multilingual-e5"} & set(emb)):
        add(wfirst(stack, "embedding"), "Your embedding (%s) was not tested; the study tested bge-m3 and "
                                        "multilingual-e5-large and found either fine." % ", ".join(emb), "ex.embedding", "scope")
    return out


# ---------------------------------------------------------------- output
QNAMES = OrderedDict([("q1", "Question type"), ("q2", "Language"), ("q3", "Evidence budget"), ("q4", "Latency"),
                      ("q5", "GPU for reranker"), ("q6", "Extra LLM calls"), ("q7", "Model may know answers"),
                      ("q8", "Persian spelling varies")])


def public(scan_out):
    """Drop internal fields (leading underscore) for JSON output."""
    def clean(v):
        if isinstance(v, dict):
            return OrderedDict((k, clean(x)) for k, x in v.items() if not str(k).startswith("_"))
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return clean(scan_out)


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return ("%d %s" % (n, u)) if u == "B" else ("%.1f %s" % (n, u))
        n /= 1024.0


def as_text(s, rec=None):
    L = []
    sc = s["scanned"]
    L.append("choose-rag project scan: " + s["root"])
    L.append("Read %d of %d files (%s). Nothing was executed and nothing was sent anywhere.%s" % (
        sc["files_read"], sc["files_seen"], human(sc["bytes_read"]),
        (" Caps hit: " + ", ".join(sc["caps_hit"]) + " (results are from a sample).") if sc["caps_hit"] else ""))
    L.append("")
    L.append("| Slot | Question | Value | Confidence | Evidence |")
    L.append("|---|---|---|---|---|")
    for q, v in s["slots"].items():
        ev = "; ".join(v["evidence"][:3]) if v["evidence"] else ""
        why = v["why"]
        L.append("| %s | %s | %s | %s | %s |" % (q, QNAMES[q], v["value"] or "-", v["confidence"] if v["applies"] else "n/a",
                                               (why.rstrip(".") + (". " + ev if ev else "")).replace("|", "/")))
    L.append("")
    L.append("Ask the user: " + (", ".join(s["ask"]) or "nothing") + ".  Confirm: " + (", ".join(s["confirm"]) or "nothing") + ".")
    c = s["corpus"]
    L.append("")
    L.append("## Corpus")
    if c["files"]:
        L.append("- %d document files, %s, about %s characters (%s)" % (c["files"], human(c["bytes"]),
                                                                         format(c["chars_estimate"], ","), c["chars_method"]))
        L.append("- types: " + ", ".join("%s %d" % kv for kv in c["types"].items()))
        L.append("- folders: " + ", ".join("%s (%d)" % kv for kv in c["folders"].items()))
        L.append("- language (%d files sampled): %s; per file: %s" % (c["sampled_files"], fmt_shares(c["script_shares"]),
                                                                       ", ".join("%s %d" % kv for kv in c["file_languages"].items()) or "-"))
        if c["pdf"]["files"]:
            L.append("- PDFs: %d, %d sampled without readable text, declared languages: %s" % (
                c["pdf"]["files"], c["pdf"]["without_readable_text"], c["pdf"]["declared_lang"] or "-"))
    else:
        L.append("- no document folders found")
    L.append("")
    L.append("## Query side")
    for k in ("questions", "ui", "prompts"):
        v = s["queries"][k]
        L.append("- %s: %d file(s), language %s%s" % (k, v["files"], v["language"],
                                                       (" (" + fmt_shares(v["script_shares"]) + "; " + ", ".join(v["examples"][:3]) + ")") if v["files"] else ""))
    L.append("")
    L.append("## Existing RAG stack")
    st = s["stack"]
    found = False
    for cat in ("vector_db", "bm25", "embedding", "reranker", "llm_selection", "fusion", "framework", "query_rewriting",
                "agent", "persian_normaliser", "retrieval_call", "long_context", "graph_or_sql", "chat_memory"):
        if st.get(cat):
            found = True
            L.append("- %s: %s" % (cat.replace("_", " "), "; ".join(first_where(st[cat], 2))))
    for cat in ("chunk_size", "chunk_overlap", "top_k", "context_budget"):
        vals = st.get(cat, [])
        if vals:
            found = True
            seen = OrderedDict()
            for e in vals:
                key = (e["name"], e["value"], e.get("unit", ""))
                seen.setdefault(key, []).append(e["where"] + (" [test]" if e["test"] else ""))
            L.append("- %s: %s" % (cat.replace("_", " "), "; ".join(
                "%s=%d%s (%s%s)" % (k[0], k[1], " " + k[2] if k[2] and k[2] != "chars" else "", ", ".join(w[:2]),
                                    " +%d" % (len(w) - 2) if len(w) > 2 else "") for k, w in list(seen.items())[:6])))
    if not found:
        L.append("- no RAG components found")
    L.append("")
    L.append("## Resources")
    r = s["resources"]
    for cat in ("gpu_infra", "gpu_code", "cpu_only", "llm_provider", "latency_fast", "latency_stream", "latency_batch"):
        if r.get(cat):
            L.append("- %s: %s" % (cat.replace("_", " "), "; ".join(first_where(r[cat], 2))))
    if not any(r.get(c) for c in r):
        L.append("- no hints")
    if s["out_of_scope"]:
        L.append("")
        L.append("## Outside the study")
        for n in s["out_of_scope"]:
            L.append("- " + n)
    if rec:
        L.append("")
        L.append("## Recommendation (choose.py, answers: %s)" % ", ".join("%s=%s" % kv for kv in rec["answers"].items()))
        L.append(rec["summary"])
        for slot_name in choose.SLOTS:
            oid = rec["recommendation"][slot_name]
            L.append("- %s: %s (`%s`)" % (slot_name, rec["cards"][oid]["text"], oid))
        L.append("")
        L.append("## Changes to this project")
        if not rec["changes"]:
            L.append("- no existing RAG stack found: build the recommended pipeline from scratch")
        for ch in rec["changes"]:
            L.append("- [%s] %s: %s Confidence: %s. Evidence: %s" % (ch["status"], ch["where"], ch["change"], ch["confidence"],
                                                                    ", ".join(ch["evidence"]) or "-"))
    return "\n".join(L)


def recommend(scan_out, overrides):
    chooser = choose.load("chooser.json")
    evidence = choose.load("evidence.json")
    raw = dict(scan_out["answers"])
    raw.update({k: v for k, v in overrides.items() if v})
    ans, state, fired = choose.choose(raw, chooser)
    rec = choose.as_json(ans, state, fired, chooser, evidence)
    rec["changes"] = plan_changes(scan_out, state, chooser, evidence, ans["q2"])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only scan of a project folder for the choose-rag chooser slots.")
    ap.add_argument("path", help="project folder to scan")
    ap.add_argument("--json", action="store_true", help="machine-readable output ('answers' feeds choose.py --json)")
    ap.add_argument("--recommend", action="store_true", help="also run choose.py and list changes for this project")
    for q in choose.QIDS:
        ap.add_argument("--" + q, help="confirmed answer for %s (overrides the scan)" % q)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = scan(a.path)
    overrides = {q: getattr(a, q) for q in choose.QIDS}
    rec = None
    if a.recommend:
        try:
            rec = recommend(s, overrides)
        except ValueError as err:
            print("error: " + str(err), file=sys.stderr)
            return 2
    if a.json:
        out = public(s)
        if rec:
            out["recommendation"] = rec
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print(as_text(s, rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
