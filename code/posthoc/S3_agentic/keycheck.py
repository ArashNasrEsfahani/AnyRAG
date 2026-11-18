import requests, json, sys
def key():
    for l in open(r"D:\KareOne\RAG Paper\.env", encoding="utf-8"):
        k, _, v = l.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
def remaining():
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key()}, timeout=20).json()["data"]
    return d.get("limit_remaining"), d.get("usage"), d.get("limit")
if __name__ == "__main__":
    print(remaining())
