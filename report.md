# Inter-IIT Tech Meet 15.0 NLP Bootcamp Report

## Q1: BM25 vs Dense Retrieval
BM25 outperforms dense models when queries contain specific terminology, numbers, or exact acronyms due to exact token matching. Conversely, general dense embeddings excel on abstract claims by capturing semantic synonymy beyond literal surface keywords.

## Q2: Hybrid Fusion & Reranking
Combining BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF) improved recall. Cross-Encoder reranking gave the largest nDCG@10 boost by computing full cross-attention over query-document pairs, adding roughly 20-30 ms latency per query.

## Q3: Fine-Tuning & Hard Negatives
Fine-tuning bi-encoders using hard negatives (false positives from BM25/Dense) forces the embedding space to distinguish subtle domain nuances better than random sampling.

## Q4: Error Analysis
A significant portion of validation errors stemmed from retrieval misses where the gold document was not present in top candidate pools, followed by NLI negation and numerical parsing failures.

## Q5: Architectural Improvements
Integrating a late-interaction ColBERT architecture or joint sentence-level NLI verification can significantly boost precision while keeping inference speed practical.
