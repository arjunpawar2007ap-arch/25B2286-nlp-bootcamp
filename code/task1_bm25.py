import os, json, numpy as np
from rank_bm25 import BM25Okapi

os.makedirs("predictions", exist_ok=True)
corpus_data = [json.loads(l) for l in open("corpus.jsonl")]
val_claims = [json.loads(l) for l in open("val_claims.jsonl")]
eval_claims = [json.loads(l) for l in open("eval_claims.jsonl")]

doc_ids, corpus_docs = [], []
for item in corpus_data:
    doc_ids.append(item["doc_id"])
    text = f"{item.get('title', '')} {item.get('abstract', item.get('text', ''))}".strip()
    corpus_docs.append(text.lower().split())

bm25 = BM25Okapi(corpus_docs, k1=1.5, b=0.75)

def run_bm25(claims, out_path, tag="bm25"):
    with open(out_path, "w") as f_out:
        for c in claims:
            cid = c.get("claim_id", c.get("id"))
            scores = bm25.get_scores(c["claim"].lower().split())
            top100 = np.argsort(scores)[::-1][:100]
            for rank, idx in enumerate(top100, 1):
                f_out.write(f"{cid} Q0 {doc_ids[idx]} {rank} {scores[idx]:.4f} {tag}\n")

run_bm25(val_claims, "predictions/val_bm25.trec")
run_bm25(eval_claims, "predictions/t1_bm25.trec")
print("Task 1 script executed successfully.")
