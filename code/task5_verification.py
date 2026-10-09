import json, torch
from collections import defaultdict
from transformers import pipeline

device_id = 0 if torch.cuda.is_available() else -1
corpus_data = [json.loads(l) for l in open("corpus.jsonl")]
val_claims = [json.loads(l) for l in open("val_claims.jsonl")]
eval_claims = [json.loads(l) for l in open("eval_claims.jsonl")]
doc_dict = {item["doc_id"]: f"{item.get('title', '')} {item.get('abstract', item.get('text', ''))}".strip() for item in corpus_data}

nli_pipe = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base", device=device_id)

def run_verif(claims, trec_path, out_csv):
    retrieved = defaultdict(list)
    for line in open(trec_path):
        parts = line.split()
        if int(parts[3]) <= 3:
            retrieved[parts[0]].append(parts[2])
    with open(out_csv, "w") as f_out:
        f_out.write("claim_id,doc_id,label\n")
        for c in claims:
            cid = str(c.get("claim_id", c.get("id")))
            for did in retrieved.get(cid, []):
                res = nli_pipe({"text": c["claim"], "text_pair": doc_dict.get(did, "")})
                lbl = res["label"].upper()
                label = "SUPPORT" if "SUPP" in lbl or "ENTAIL" in lbl else ("CONTRADICT" if "CONTR" in lbl or "REFUT" in lbl else "NEI")
                if label != "NEI":
                    f_out.write(f"{cid},{did},{label}\n")

run_verif(eval_claims, "predictions/t3_rerank.trec", "predictions/t5_verification.csv")
run_verif(val_claims, "predictions/val_bm25.trec", "predictions/t5_val_verification.csv")
print("Task 5 script executed successfully.")
