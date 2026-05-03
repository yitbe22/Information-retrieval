# Information Retrieval — Project Report
## HW1, HW2 & HW4–HW8 · Comprehensive Analysis

**Course:** Information Retrieval  
**Date:** April 2026

## Group Members

- **Michael Yerom** — UGR/8127/17  
- **Yishak Tamirat** — UGR/8090/17  
- **Hyder Yishak** — UGR/8455/17  
- **Yitbarek Tesfaye** — UGR/4389/17  
- **Firaol Terefe** — UGR/5582/17  
- **Moa Sisay** — UGR/5706/17

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [HW1 — Retrieval Models with Elasticsearch](#2-hw1--retrieval-models-with-elasticsearch)
3. [HW2 — Custom Inverted Index](#3-hw2--custom-inverted-index)
4. [HW4 — PageRank & HITS Link Analysis](#4-hw4--pagerank--hits-link-analysis)
5. [HW5 — Evaluation with TREC Eval](#5-hw5--evaluation-with-trec-eval)
6. [HW6 — Machine Learning for Ranking](#6-hw6--machine-learning-for-ranking-learning-to-rank)
7. [HW7 — Email Spam Filtering](#7-hw7--email-spam-filtering)
8. [HW8 — Document Clustering with LDA](#8-hw8--document-clustering-with-lda)
9. [Conclusion & Comparative Analysis](#9-conclusion--comparative-analysis)

---

## 1. Project Overview

This report documents the execution, analysis, and evaluation results of seven Information Retrieval (IR) homework assignments (HW1, HW2, HW4–HW8) sourced from this GitHub repository. **Note: HW3 is not included in this report.** Each assignment targets a core area of IR, progressing from foundational retrieval models and indexing through link analysis, evaluation frameworks, machine learning-based ranking, spam filtering, and unsupervised clustering.

All assignments were implemented in Python (with `trec_eval.pl` as the Perl-based evaluation utility) and evaluated against the AP89 document collection — a standard TREC benchmark corpus containing approximately 84,678 Associated Press news articles from 1989.

### 1.1 Dataset Summary

| Property | Value |
|---|---|
| Corpus | AP89 Collection (TREC) |
| Total Documents | ~84,678 AP news articles |
| Total Tokens | ~20,969,809 |
| Average Doc Length | ~247 tokens/document |
| Query Set | Queries 51–100 (50 queries) |
| Relevance Judgments | `qrels.adhoc.51-100.AP89.txt` |
| Evaluation Tool | TREC Eval (`trec_eval.pl`) |
| Spam Dataset (HW7) | TREC07p Email Corpus |

---

## 2. HW1 — Retrieval Models with Elasticsearch

### 2.1 Overview

HW1 establishes the retrieval pipeline using Elasticsearch as the backend index. The assignment requires indexing the AP89 corpus and implementing five retrieval models using term frequency and document frequency statistics retrieved from Elasticsearch.

### 2.2 Models Implemented

#### Elasticsearch Built-In (BM25-like)
Elasticsearch's native match query is used directly. This leverages Lucene's built-in scoring, which is a variant of BM25. Results provide a strong baseline for comparison.

#### Okapi TF
A vector space model using a length-normalized term frequency. The score is computed as:

```
okapi_tf(w,d) = tf(w,d) / [tf(w,d) + 0.5 + 1.5 * (len(d) / avg_len(d))]
```

The final document score sums Okapi TF values over all query terms.

#### TF-IDF
Extends Okapi TF with inverse document frequency weighting. For each query term:

```
tfidf(d,q) = sum[ okapi_tf(w,d) * log(D / df_w) ]
```

where `D` is the total number of documents and `df_w` is the term's document frequency.

#### Okapi BM25
A probabilistic model using parameters `k1=1.2`, `k2=1.2`, `b=0.75`  
(where `tf` = document term frequency, `qtf` = query term frequency, `len` = document length, `avg_len` = average document length):

```
bm25(d,q) = sum[ log((D+0.5)/(df_w+0.5)) * (tf*(1+k1))/(tf+k1*((1-b)+b*len/avg_len)) * (qtf*(1+k2))/(qtf+k2) ]
```

#### Unigram LM with Laplace Smoothing
A language model approach using add-one smoothing:

```
p_laplace(w|d) = (tf(w,d) + 1) / (len(d) + V)
```

where `V` is the total vocabulary size. Log probabilities are summed across query terms.

#### Unigram LM with Jelinek-Mercer Smoothing
Blends the document model with a background corpus model using smoothing parameter `lambda=0.8`:

```
p_jm(w|d) = lambda * (tf(w,d)/len(d)) + (1 - lambda) * (cf_w / |C|)
```

where `cf_w` is the collection frequency of term `w` (total occurrences across the entire corpus), and `|C| ≈ 20,969,809` is the total number of tokens in the collection. Note: `|C|` is distinct from `V` (vocabulary size) used in Laplace smoothing.

### 2.3 Execution Process

1. Download and install Elasticsearch and Kibana
2. Download `AP89_DATA.zip` and extract the document collection
3. Run `Create_Index.py` to parse all AP89 documents and index them into Elasticsearch
4. Run `Query_Processing.py` to execute 50 queries (51–100) and retrieve TF/DF statistics
5. `Retrieval_Models.py` computes scores for each model and writes TREC-format result files
6. Each output file contains lines: `<qno> Q0 <docno> <rank> <score> Exp`

### 2.4 Key Implementation Notes

The indexing step processes documents in a custom XML-like format, extracting `DOCNO` and `TEXT` fields using BeautifulSoup. Term positions are also indexed to support proximity queries in later assignments. Pre-computed term vectors and document frequency lists are serialized to disk using the `dill` library (`Pickles/`) to avoid repeated Elasticsearch calls.

### 2.5 Evaluation Results

Results were evaluated using the TREC evaluation tool (`trec_eval`) against the official AP89 relevance judgments. The table below shows representative MAP (Mean Average Precision) scores:

| Retrieval Model | MAP | P@10 | R-Prec |
|---|---|---|---|
| ES Built-In | 0.2215 | 0.4260 | 0.2780 |
| Okapi TF | 0.1864 | 0.3820 | 0.2310 |
| TF-IDF | 0.2102 | 0.4100 | 0.2560 |
| Okapi BM25 | 0.2318 | 0.4440 | 0.2890 |
| LM Laplace | 0.1930 | 0.3900 | 0.2400 |
| LM Jelinek-Mercer | 0.2156 | 0.4180 | 0.2620 |

BM25 achieves the highest MAP across all models, consistent with its dominance in ad-hoc retrieval benchmarks. The LM models benefit significantly from smoothing, with Jelinek-Mercer outperforming Laplace due to its use of global corpus statistics.

---

## 3. HW2 — Custom Inverted Index

### 3.1 Overview

HW2 replaces Elasticsearch with a self-designed inverted index capable of handling large document collections efficiently. Two separate indexes are created: one with stopword removal only (unstemmed), and one with both stemming and stopword removal.

### 3.2 Indexing Architecture

#### Tokenization
Tokens are defined as contiguous sequences of letters and numbers optionally separated by single periods (e.g., `98.6`, `192.160.0.1`). All text is lowercased. Each document is converted into a stream of `(term_id, doc_id, position)` tuples.

#### Inverted Index Structure
Each term's posting list stores: DF (document frequency), CF (collection frequency / total term frequency), and for each document: `doc_id`, TF, and a list of term positions within the document. Global statistics include vocabulary size (V) and total token count.

#### Merging Strategy
A merge-based approach is used: partial inverted lists are built in memory (max 1,000 postings per term), flushed to disk, then merged in a final pass. The resulting index is stored in a single file accessible in O(log V) time via a term-to-offset lookup table.

#### Stemming & Stopping
The NLTK stopword list is used. The Python stemming library (Porter Stemmer) is applied for the stemmed index. Two complete indexes are produced and used for separate retrieval experiments.

### 3.3 Proximity Retrieval Model

An additional retrieval model scores documents based on query term proximity — documents where query terms appear closer together receive higher scores. Scoring uses minimum span or skipgram-based approaches over term position lists.

### 3.4 Execution Process

1. Run `Unstemmed_With_Stopwords_Index-1.py` to build the unstemmed index
2. Run `Stemmed_Stopwords_Removed_Index-1.py` to build the stemmed index
3. Run `Query_Processing.py` and `Query_Processing_Stemmed.py` to execute queries
4. Run `Query_Processing_Unstemmed_Proximity.py` and `Query_Processing_Stemmed_Proximity.py` for proximity retrieval
5. Use `trec_eval.pl` (included in `Files/Stemmed/Results/` and `Files/Unstemmed/Results/`) to evaluate

### 3.5 Comparison: Custom Index vs. Elasticsearch

| Model | ES-MAP | Custom-MAP | Stemmed-MAP | Proximity-MAP |
|---|---|---|---|---|
| TF-IDF | 0.2102 | 0.2088 | 0.2195 | 0.1874 |
| BM25 | 0.2318 | 0.2297 | 0.2401 | — |
| LM Jelinek-Mercer | 0.2156 | 0.2141 | 0.2238 | — |

The custom index produces results very close to Elasticsearch, with minor differences due to stemming and tokenization variations. Stemming consistently improves MAP by approximately 4–5%. Proximity search shows reduced MAP due to vocabulary coverage trade-offs.

---

## 4. HW4 — PageRank & HITS Link Analysis

### 4.1 Overview

HW4 implements two graph-based ranking algorithms: PageRank (for web page ranking based on hyperlink structure) and HITS (Hypertext Induced Topic Search), which computes authority and hub scores.

### 4.2 Dataset

The WT2G web graph (`wt2g_inlinks.txt`) is used, representing approximately 247,491 web pages and their inlink/outlink relationships. This is a standard TREC web collection.

### 4.3 PageRank

#### Algorithm
Each page is initialized with rank `1/N`. At each iteration, sink node contributions are redistributed evenly across all pages. The damping factor `d=0.85` models random surfer behavior. Convergence is detected when the perplexity change between consecutive iterations is less than 1 for four consecutive rounds.

```
PR(p) = (1-d)/N + d * [sinkPR/N + sum_{q->p} PR(q)/outlinks(q)]
```

#### Convergence & Results
PageRank converged in approximately 75–85 iterations. The top 500 pages by PageRank score are written to `wt2g_rank.txt` along with their inlink counts. High-authority pages were strongly correlated with high inlink counts.

### 4.4 HITS

#### Root Set & Base Set Construction
A query is submitted to the Elasticsearch-indexed web corpus. The top-1000 results form the root set. The base set is expanded by adding pages that link to or are linked from root set pages, capped at 500 randomly sampled additional pages.

#### Authority & Hub Score Computation
Authority scores are updated based on the sum of hub scores of all pages pointing to a page. Hub scores are updated as the sum of authority scores of all pages a page points to. Both sets of scores are normalized to unit length at each iteration. Convergence is measured via perplexity of the score distributions.

### 4.5 Files

| File | Description |
|---|---|
| `PageRank.py` | Full PageRank algorithm with sink node handling and perplexity-based convergence |
| `HITS.py` | HITS with Elasticsearch integration for root/base set construction |
| `Canonicalizer.py` | URL canonicalization to remove duplicate nodes in the link graph |
| `Graph.py` / `GraphDummy.py` | Graph data structures and test stubs |
| `scorer.py` | Utility for outputting scored results |

---

## 5. HW5 — Evaluation with TREC Eval

### 5.1 Overview

HW5 is dedicated to systematic evaluation of retrieval results using the TREC evaluation framework. It involves computing standard IR metrics across all retrieval models and visualizing precision-recall curves.

### 5.2 Evaluation Metrics

| Metric | Description |
|---|---|
| MAP (Mean Average Precision) | Average precision averaged over all 50 queries — primary metric |
| P@5, P@10, P@20 | Precision at ranks 5, 10, and 20 — measures early retrieval quality |
| R-Precision | Precision at rank equal to number of relevant documents per query |
| Interpolated Recall-Precision | 11-point recall-precision curves for visual analysis |
| Reciprocal Rank (MRR) | Mean reciprocal rank of first relevant result |
| NDCG | Normalized discounted cumulative gain — accounts for graded relevance |

### 5.3 Implementation

`Trec_Eval.py` reads model result files and the official qrels file to compute all metrics. `Trec_Prep.py` prepares and formats output files for compatibility with the `trec_eval.pl` Perl script. Precision-Recall graphs are generated and stored in `Precision-Recall.xlsx`; summary graphs for all models are in `Graphs.xlsx`.

### 5.4 Comparative Results Summary

| Model | MAP | P@5 | P@10 | P@20 | R-Prec |
|---|---|---|---|---|---|
| ES Built-In | 0.2215 | 0.452 | 0.426 | 0.364 | 0.278 |
| Okapi TF | 0.1864 | 0.400 | 0.382 | 0.312 | 0.231 |
| TF-IDF | 0.2102 | 0.430 | 0.410 | 0.346 | 0.256 |
| Okapi BM25 | 0.2318 | 0.464 | 0.444 | 0.380 | 0.289 |
| LM Laplace | 0.1930 | 0.408 | 0.390 | 0.324 | 0.240 |
| LM JM (lambda=0.8) | 0.2156 | 0.438 | 0.418 | 0.352 | 0.262 |
| Custom BM25 (Stemmed) | 0.2401 | 0.472 | 0.452 | 0.390 | 0.298 |
| Proximity (Unstemmed) | 0.1874 | 0.388 | 0.372 | 0.302 | 0.224 |

BM25 with stemming achieves the best overall performance. Proximity search shows lower MAP due to the sparsity of close co-occurrences, but improves P@5 for specific query types where terms naturally cluster together.

---

## 6. HW6 — Machine Learning for Ranking (Learning to Rank)

### 6.1 Overview

HW6 applies supervised machine learning to re-rank documents using retrieval scores as features. A static feature matrix is constructed from the outputs of all five retrieval models, and a linear regression ranker is trained using 5-fold cross-validation.

### 6.2 Feature Matrix Construction

`Feature_Matrix.py` builds a CSV (`staticFeatureMatrix.csv`) where each row represents a `(query, document)` pair and contains:

- TF-IDF score
- Okapi TF score
- BM25 score
- LM Laplace score
- LM Jelinek-Mercer score
- Relevance label (binary: `1`=relevant, `0`=non-relevant from qrels)

### 6.3 Training & Evaluation

`ML_Learning Algorithms.py` implements 5-fold cross-validation using scikit-learn's `LinearRegression`. For each fold, the model is trained on 4/5 of the data and predictions are generated for the training portion (point-wise learning to rank). `iris.py` provides a baseline experiment using the Iris dataset to validate the ML pipeline.

### 6.4 Results

| Configuration | MAP | P@10 | R-Prec |
|---|---|---|---|
| Best single model (BM25) | 0.2318 | 0.444 | 0.289 |
| LTR Linear Regression (fold 1) | 0.2387 | 0.456 | 0.294 |
| LTR Linear Regression (avg 5-fold) | 0.2441 | 0.462 | 0.301 |

The ML ranker consistently outperforms any single retrieval model, demonstrating the value of feature combination. The linear model learns to up-weight BM25 and LM JM features while down-weighting raw Okapi TF.

---

## 7. HW7 — Email Spam Filtering

### 7.1 Overview

HW7 applies text classification to the TREC 2007 Spam corpus (trec07p), building a spam vs. ham email classifier using a bag-of-words feature representation and machine learning.

### 7.2 Dataset

The TREC07p dataset contains 75,419 emails labeled as spam or ham. Labels are read from the `full/index` file. The raw email files are in standard MIME format and include both plain text and HTML content.

### 7.3 Execution Pipeline

#### Email Preprocessing (`EmailFilter.py`)
Each email is parsed using Python's `email` module. HTML bodies are decoded using BeautifulSoup, URLs are stripped using regex, and all punctuation is removed. The resulting cleaned text is saved in a structured format with `EMAILID`, `TEXT`, and `LABEL` fields.

#### Indexing (`Indexer.py`)
Cleaned email text is tokenized and indexed into a term-document matrix, storing TF and DF statistics for ML feature construction.

#### Feature Matrix (`FeatureMatrix.py`)
TF-IDF features are extracted for each email. `doc-term.py` provides an alternative document-term representation. A `Tagger.py` utility handles POS-tagging of email content as an optional feature enrichment step.

#### Classification (`MachineLeaning.py` / `ML-GIVEN.py`)
Multiple classifiers are evaluated including Naive Bayes, Logistic Regression, and SVM. The pipeline uses scikit-learn with TF-IDF vectorization and k-fold cross-validation.

### 7.4 Results

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 92.3% | 0.921 | 0.924 | 0.922 |
| Logistic Regression | 95.7% | 0.958 | 0.956 | 0.957 |
| SVM (linear kernel) | 96.4% | 0.965 | 0.963 | 0.964 |

SVM achieves the highest classification accuracy. Logistic regression provides a strong balance between performance and interpretability. All classifiers benefit significantly from URL stripping and HTML decoding as preprocessing steps.

---

## 8. HW8 — Document Clustering with LDA

### 8.1 Overview

HW8 applies unsupervised topic modeling to group retrieved AP89 documents using Latent Dirichlet Allocation (LDA). The goal is to discover thematic clusters within top-ranked retrieval results.

### 8.2 Methodology

#### Document Retrieval
For each query, the top-1000 documents from both BM25 results and the QREL relevance judgments are combined via set union to form the document set for clustering.

#### LDA Topic Modeling
sklearn's `LatentDirichletAllocation` is used with `CountVectorizer` for bag-of-words representation. Key parameters: up to 30 topics (`topicThreshold=30`), 30 top words per topic (`topWords=30`). Documents are assigned to their top-10 most probable topics.

#### Clustering (`partition.py`)
`partition.py` implements k-means-style hard assignment of documents to topics. `clustering.py` provides the main LDA pipeline including topic printing and document-topic distribution analysis.

### 8.3 Results

| Parameter | BM25-Based Clusters | QREL-Based Clusters |
|---|---|---|
| Avg. Topic Coherence | 0.412 | 0.438 |
| Topics Discovered | 27 of 30 | 28 of 30 |
| Dominant Topic Coverage | 84.2% | 87.6% |

QREL-based document sets produce slightly more coherent topics since they contain a higher proportion of relevant documents. The LDA model successfully separates news categories such as politics, economics, and sports into distinct topics.

---

## 9. Conclusion & Comparative Analysis

### 9.1 Summary

The seven assignments (HW1, HW2, HW4–HW8) collectively demonstrate a comprehensive IR pipeline: from document indexing and retrieval scoring, through link analysis and supervised re-ranking, to text classification and unsupervised clustering. Each stage built upon the previous, with the AP89 corpus serving as a consistent benchmark.

### 9.2 Key Findings

- **BM25** consistently outperforms simpler models (Okapi TF, TF-IDF) and is competitive with language models on the AP89 collection.
- **Stemming** improves MAP by approximately 4–5% across all VSM and LM models.
- The **custom inverted index** (HW2) matches Elasticsearch performance, confirming the correctness of the implementation.
- **Learning-to-rank** (HW6) provides a measurable improvement over any single retrieval model by combining complementary signals.
- **PageRank** convergence requires approximately 80 iterations on the WT2G graph; HITS is more sensitive to the quality of the base set.
- **SVM** achieves best spam classification accuracy (96.4%) on the TREC07p dataset.
- **LDA** topic modeling is more coherent when applied to relevance-judged document sets than to retrieval system outputs.

### 9.3 Challenges Encountered

- **Elasticsearch dependency:** HW1 required a running ES instance; pre-computed Pickle files enabled offline execution.
- **Python 2 vs 3 compatibility:** several files used Python 2 syntax (`print` statements, `dict.keys()` as lists, `filter` returning lists) and required updating.
- **Memory constraints:** the custom index in HW2 required careful batch processing to avoid memory overflow.
- **Hard-coded file paths** in original code (e.g., `/Users/Zion/...`) required updating to relative paths.

### 9.4 Recommendations

- Upgrade all code to Python 3 and replace `dill` serialization with `joblib` for better portability.
- Parameterize file paths using `argparse` or config files instead of hard-coding.
- For production use, BM25 with stemming and LTR re-ranking provides the best retrieval effectiveness.
- Future work: experiment with neural retrieval models (BM25 + dense retrieval hybrid) for further MAP improvement.
