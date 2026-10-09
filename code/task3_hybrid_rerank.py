import json, torch
from collections import defaultdict
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
corpus_data = [json.loads(l) for l in open("corpus.jsonl")]
eval_claims = [json.loads(l) for l in open("eval_claims.jsonl")]
doc_dict = {item["doc_id"]: f"{item.get('title', '')} {item.get('abstract', item.get('text', ''))}".strip() for item in corpus_data}

def rrf(t1, t2, out_path, k=60):
    rankings = defaultdict(lambda: defaultdict(float))
    for file_path in [t1, t2]:
        for line in open(file_path):
            parts = line.split()
            rankings[parts[0]][parts[2]] += 1.0 / (k + int(parts[3]))
    with open(out_path, "w") as f_out:
        for cid, doc_scores in rankings.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            for rank, (did, score) in enumerate(sorted_docs, 1):
                f_out.write(f"{cid} Q0 {did} {rank} {score:.6f} hybrid\n")

rrf("predictions/t1_bm25.trec", "predictions/t2_general.trec", "predictions/t3_hybrid.trec")

reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)
candidates = defaultdict(list)
for line in open("predictions/t3_hybrid.trec"):
    parts = line.split()
    candidates[parts[0]].append(parts[2])

with open("predictions/t3_rerank.trec", "w") as f_out:
    for c in eval_claims:
        cid = str(c.get("claim_id", c.get("id")))
        pool = candidates.get(cid, [])[:100]
        top30, rest = pool[:30], pool[30:]
        pairs = [[c["claim"], doc_dict.get(did, "")] for did in top30]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(top30, scores), key=lambda x: x[1], reverse=True)
        rank = 1
        for did, score in reranked:
            f_out.write(f"{cid} Q0 {did} {rank} {float(score):.4f} rerank\n")
            rank += 1
        for did in rest:
            f_out.write(f"{cid} Q0 {did} {rank} 0.0000 rerank\n")
            rank += 1
print("Task 3 script executed successfully.")
