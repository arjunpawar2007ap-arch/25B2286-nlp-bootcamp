import os, json, torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
corpus_data = [json.loads(l) for l in open("corpus.jsonl")]
eval_claims = [json.loads(l) for l in open("eval_claims.jsonl")]
doc_dict = {item["doc_id"]: f"{item.get('title', '')} {item.get('abstract', item.get('text', ''))}".strip() for item in corpus_data}
doc_ids = [item["doc_id"] for item in corpus_data]
texts = [doc_dict[did] for did in doc_ids]

def run_dense_retrieval(model_name, claims, out_path, tag):
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

run_dense_retrieval("sentence-transformers/all-MiniLM-L6-v2", eval_claims, "predictions/t4_random.trec", "ft_random")
run_dense_retrieval("sentence-transformers/all-MiniLM-L6-v2", eval_claims, "predictions/t4_hard.trec", "ft_hard")
print("Task 4 script executed successfully.")
