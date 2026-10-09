import json, torch
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"
corpus_data = [json.loads(l) for l in open("corpus.jsonl")]
val_claims = [json.loads(l) for l in open("val_claims.jsonl")]
eval_claims = [json.loads(l) for l in open("eval_claims.jsonl")]

doc_ids, texts = [], []
for item in corpus_data:
    doc_ids.append(item["doc_id"])
    texts.append(f"{item.get('title', '')} {item.get('abstract', item.get('text', ''))}".strip())

def run_dense(model_name, claims, out_path, tag):
    model = SentenceTransformer(model_name, device=device)
    doc_emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
    claim_texts = [c["claim"] for c in claims]
    claim_ids = [c.get("claim_id", c.get("id")) for c in claims]
    claim_emb = model.encode(claim_texts, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(claim_emb, doc_emb, top_k=100)
    with open(out_path, "w") as f_out:
        for cid, hit_list in zip(claim_ids, hits):
            for rank, hit in enumerate(hit_list, 1):
                f_out.write(f"{cid} Q0 {doc_ids[hit['corpus_id']]} {rank} {hit['score']:.4f} {tag}\n")

run_dense("BAAI/bge-base-en-v1.5", val_claims, "predictions/val_general.trec", "general")
run_dense("BAAI/bge-base-en-v1.5", eval_claims, "predictions/t2_general.trec", "general")
run_dense("allenai/scibert_scivocab_uncased", eval_claims, "predictions/t2_scientific.trec", "scientific")
print("Task 2 script executed successfully.")
