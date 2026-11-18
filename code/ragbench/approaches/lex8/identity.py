"""LEX8 m3 artifact identity (Astra R1): an index-time bge-m3 artifact is usable only for the exact rows it was built
from. The LEX7 check (row count + a doc-id checksum) cannot see changed text or rows reordered inside one document.

manifest = {n, texts_sha1 (the exact encoded texts, in order), rows_sha1 ((doc_id, start, end) per row, in order),
            encoder {model, revision (HF snapshot commit), max_length, fp16, m3_version}, preprocess (index-text
            builder name + version; query normalisation), kind}
check(manifest, texts, rows, encoder) raises IdentityError naming every mismatching field.
"""
import hashlib
import json

IDENTITY_VERSION = "lex8-identity-v1"


class IdentityError(RuntimeError):
    pass


def sha_texts(texts):
    h = hashlib.sha1()
    for t in texts:
        b = (t or "").encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def sha_rows(rows):
    h = hashlib.sha1()
    for d, a, b in rows:
        h.update(("%s\t%d\t%d\n" % (d, int(a), int(b))).encode("utf-8"))
    return h.hexdigest()


def encoder_info(model, revision, max_length, fp16, m3_version):
    return {"model": model, "revision": revision, "max_length": int(max_length), "fp16": bool(fp16),
            "m3_version": m3_version}


def manifest(kind, texts, rows, encoder, preprocess):
    texts, rows = list(texts), list(rows)
    if len(texts) != len(rows):
        raise ValueError("texts and rows differ in length")
    return {"identity_version": IDENTITY_VERSION, "kind": kind, "n": len(texts), "texts_sha1": sha_texts(texts),
            "rows_sha1": sha_rows(rows), "encoder": dict(encoder), "preprocess": dict(preprocess)}


def check(man, texts, rows, encoder=None, preprocess=None):
    """-> None, or raise IdentityError listing every mismatch (n, texts, rows, encoder fields, preprocess)."""
    bad = []
    texts, rows = list(texts), list(rows)
    if man.get("n") != len(texts):
        bad.append("n %s != %d" % (man.get("n"), len(texts)))
    else:
        if man.get("texts_sha1") != sha_texts(texts):
            bad.append("texts_sha1")
        if man.get("rows_sha1") != sha_rows(rows):
            bad.append("rows_sha1")
    if encoder is not None:
        for k, v in encoder.items():
            if k == "fp16":
                continue                      # fp16 vs fp32 query encoding is allowed; the artifact records its own
            if (man.get("encoder") or {}).get(k) != v:
                bad.append("encoder.%s %r != %r" % (k, (man.get("encoder") or {}).get(k), v))
    if preprocess is not None and man.get("preprocess") != preprocess:
        bad.append("preprocess %s != %s" % (json.dumps(man.get("preprocess")), json.dumps(preprocess)))
    if bad:
        raise IdentityError("m3 artifact identity mismatch (%s): %s" % (man.get("kind"), "; ".join(bad)))
