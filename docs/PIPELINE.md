# Pipeline details

`notebooks/Pipeline.ipynb` runs the whole SCEDG pipeline for one dataset split. This page follows the notebook section by section and describes what each stage computes, which settings control it, and which files it writes. Section and equation numbers refer to the paper.

All intermediate files of a split are written to

```
CACHE_DIR = /content/kg_vqa_cache/<okvqa|aokvqa>/<train|validation|test>
```

Stages 5, 6, and 6.1 append one JSON line per question and skip questions that are already in their output file, so a stopped run can be resumed by running the notebook again.

## 1 to 3. Environment and configuration

The first cells mount Google Drive, install the pinned packages, clone [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) and [SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark), create the SGG-Benchmark virtual environment, and download VG150. **3. Configuration** defines every path and hyperparameter. The variables you are most likely to change are

| Variable | Default | Meaning |
|---|---|---|
| `ROOT_DIR` | `/content/drive/MyDrive/dataset` | persistent storage for the knowledge base and models |
| `DATASET_NAME` | `"OK-VQA"` | `"OK-VQA"` or `"A-OKVQA"` |
| `DATASET_SPLIT` | `"train"` | `"train"`, `"validation"`, or `"test"` |
| `SEED` | `42` | random seed, also used for the 85/15 OK-VQA split |
| `HF_USERNAME` | `"your-hf-username"` | Hugging Face account that receives the adapters and logs |
| `REACT_RUN_DIR` | `<ROOT_DIR>/models/react_yoloworldv2_vg150` | REACT relation checkpoint |

`RUN_NAME = "<dataset>_qwen_k_6m_mambancd_rerank_run1"` names the experiment. The fine-tuned adapter is stored in the public model repository `MODEL_REPO_ID = <HF_USERNAME>/<RUN_NAME>`, and the logs in the private repository `TRACKING_REPO_ID = <HF_USERNAME>/kg-vqa_experimental_results`.

## 4. Dataset loading and multi-view query formation (Sec. 3.2)

1. The split is loaded from the Hugging Face Hub (see [DATA.md](DATA.md)).
2. **Scene graphs.** Unique images are exported to `CACHE_DIR/react_images/`, and REACT with the YOLO-WorldV2 backbone is run once in the SGG-Benchmark environment. The predictions are stored in `CACHE_DIR/react_sgg/custom_prediction.json` and `custom_data_info.json`. When these files exist, REACT is not run again.
3. **Contextual view.** The REACT triples are joined into a caption. The contextual queries are the question, the caption, and their concatenation.
4. **Visual view.** GLiNER (threshold 0.4) and spaCy noun chunks extract concepts from the question. Each concept is matched to the most similar scene-graph entity with MiniLM, and the matched boxes are cropped with 5 pixels of padding (crops smaller than 20 pixels are dropped). The visual queries are the full image and these crops.
5. **Entity-anchor view.** Noun chunks and named entities of the question form `N_q`. Every scene-graph entity is aligned to a ConceptNet node with MiniLM and expanded by one hop. The two nodes most similar to the question form `E_2`. ConceptNet responses are cached in `CACHE_DIR/conceptnet_minilm_cache.json`.

## 5. Multi-view retrieval and RRF (Sec. 3.2, Eq. 4)

Images and texts are encoded with LongCLIP-L. Each query searches all FAISS chunks and keeps its own top `RETRIEVAL_TOP_K_PER_QUERY = 120` passages. The ranked lists of all views are fused with reciprocal rank fusion, `sum 1 / (RRF_K + rank)` with `RRF_K = 60`, and the top `RAW_FUSED_TOP_K = 350` candidates are kept.

Output `raw_retrieval.jsonl`, one line per question:

```json
{"question_id": ..., "image_id": ..., "question_text": "...", "caption": "...",
 "context_queries": [...], "visual_concepts": [...], "visual_alignment": [...],
 "question_entity_anchors": [...], "conceptnet_top2": [...], "anchor_queries": [...],
 "scene_graph_triples": [...], "num_query_views": ...,
 "raw_candidates": [{"wikidata_id": "Q...", "content": "...", "score": ..., "rrf_score": ..., "view_hits": ...}, ...]}
```

## 6. Cross-encoder reranking (Sec. 3.3)

`mixedbread-ai/mxbai-rerank-large-v1` scores each candidate against `"<question> Context: <caption>"` and the top `RERANK_TOP_K = 50` are kept. When `Wiki6M_ver_1_0.jsonl.gz` is available, the full Wikipedia summary replaces the 300-character index text.

Output `cross_encoder_top50.jsonl` with `retrieved_knowledge` (passage texts) and `retrieved_metadata` (`content`, `wikidata_id`, `rerank_score`, `origin_score`).

## 6.1. Semantic-compressive evaluator (Sec. 3.3, Eq. 5 to 9)

The code calls this stage *Mamba gating*. It does not apply the confidence gate.

For the top `MAMBA_INPUT_TOP_M = 25` cross-encoder candidates, a frozen `state-spaces/mamba2-130m` reads the query `"Question: <q> Context: <caption> Answer:"`, each passage, and their concatenation (at most 512 tokens each). The representation energy `E` of a sequence is the mean over its tokens of the mean squared value of the backbone output, ignoring padding. The Mamba similarity is

```
1 - clip((E(u + k) - min(E(u), E(k))) / max(E(u), E(k)), 0, 1)
```

and the SCE score is `W_MXBAI * sigmoid(rerank_score) + W_MAMBA * mamba_similarity` with `W_MXBAI = W_MAMBA = 0.5`. The top `FINAL_KNOWLEDGE_TOP_K = 10` candidates are kept.

Output `mamba_top10.jsonl`. Each entry of `retrieved_metadata` also stores `fusion_score` and its two components.

## 7. Summary dataset

The ten selected passages are joined with the question and the reference answers. The training label is the most frequent reference answer that appears in the passages, or the most frequent answer when none appears. On `train`, questions without answers are skipped and the share of questions with an answer-containing passage is printed.

Output `summary_dataset.json`, a list of `{question_id, image_id, question_text, caption, knowledge_texts, label, answers_list}`.

## 8. Task-specific reranker (Sec. 3.4, Eq. 10 and 11)

In the code, the task-specific reranker is called the fine-tuned cross-encoder.

- **Targets** (`train` only). A passage gets the hard label 1 when a normalized reference answer appears in it as a whole word. When none of the ten passages of a question has a hard positive, the frozen teacher (`mxbai-rerank-large-v1`) scores `"Context: <caption>. Question: <q>. Answer: <label>"` against each passage, and the soft target is `sigmoid(TEACHER_SCALE * logit)` with `TEACHER_SCALE = 3.0`. Output `cross_encoder_labels.json`.
- **Training** (`train` only). The reranker is initialized from `mxbai-rerank-large-v1` with a single output logit. Each question contributes its ten passages, paired with `"Context: <caption>. Question: <q>"` (at most 350 tokens), and the loss is binary cross-entropy over the real passages. Settings: 8 epochs, learning rate 1e-4, batch 8 with gradient accumulation 8, 50 warmup steps, bf16 when available, gradient checkpointing. The weights are saved to `<ROOT_DIR>/models/<dataset>/cross_encoder/mixedbread/cross_encoder_final_weights.pth`.
- **Top-1 selection** (all splits). The trained reranker scores the ten passages of every question and keeps the best one together with its probability `S_max`. Output `dataset_one_knowledge.json` with `knowledge_texts` (the selected passage) and `best_knowledge_prob` (`S_max`).

`evaluate_cross_encoder(threshold)` is an optional diagnostic and is not called by default.

## 9. Qwen2.5-VL LoRA fine-tuning (Sec. 3.6, Algorithm 1)

Runs only on `train`. A question receives its passage when `S_max >= TRAIN_PROB_THRESHOLD` (0.7). The prompt is

```
context: <passage>
question: <question>
```

or `question: <question>` without a passage. The loss covers only the assistant answer.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct`, bf16 when available, SDPA attention |
| LoRA | rank 32, alpha 64, dropout 0.2, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimization | 6 epochs, learning rate 5e-6, cosine schedule, 10% warmup, weight decay 0.05 |
| Batch | 8 per device with gradient accumulation 4, maximum length 2,048 tokens |
| Checkpoints | every 100 steps, at most 4 kept |
| Early stop | training stops at a save step after half of the planned steps once the logged loss is 0.3 or lower |

The run resumes from the latest checkpoint found in `MODEL_REPO_ID`. The final adapter and processor are uploaded to `MODEL_REPO_ID`, and `train_loss.jsonl`, `train_summary_metadata.json`, and TensorBoard logs are uploaded to `TRACKING_REPO_ID`.

## 10. Gated inference and evaluation (Sec. 3.5, Eq. 12)

For each question, the gate adds the selected passage to the prompt when `S_max >= threshold`. Answers are generated greedily with at most 128 new tokens.

| Split | Adapters evaluated | Thresholds |
|---|---|---|
| `validation` | every checkpoint in `MODEL_REPO_ID` and the final adapter | 0.55, 0.7, 0.8, 0.9, 1.0 |
| `test` | final adapter | 0.9 |

VQA accuracy is `min(#matching reference answers / 3, 1)` after lowercasing and removing punctuation and articles. Each evaluation appends one line to `/content/experiment_logs/<RUN_NAME>/eval_results_tracker.jsonl`:

```json
{"dataset": "OK-VQA", "split": "test", "strategy_name": "okvqa_final_rag_0.9",
 "checkpoint_used": "final_model", "test_prob_threshold": 0.9, "dataset_size": 5046, "eval_batch_size": 32,
 "accuracy_vqa": ..., "avg_llm_gen_sec": ..., "avg_e2e_latency_sec": ...,
 "throughput_samples_per_sec": ..., "total_eval_time_mins": ...}
```

and writes `predictions_<strategy>.json` with `question_id`, `image_id`, `question_text`, `full_context`, `answers_list`, `extracted_answer`, and `accuracy_score` for every question. The latency fields cover answer generation over the cached evidence only, not retrieval and filtering.

## Preparation notebook

`notebooks/Preparation.ipynb` builds the retrieval index once. See [DATA.md](DATA.md#faiss-index-built-by-preparationipynb) for the file format. Settings: LongCLIP-L text encoder, batch size 1,024, 512,000 vectors per chunk, resumable through `index_progress.json`.
