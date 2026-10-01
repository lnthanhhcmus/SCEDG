# Data and external resources

This page lists every dataset, knowledge source, and pretrained model that the notebooks use, where each one comes from, and where the notebooks expect to find it. Each resource is distributed under its own license. Please check the original source before redistributing data or models.

## Benchmarks

### OK-VQA

- Official page: <https://okvqa.allenai.org>
- Used through the Hugging Face mirrors [`Multimodal-Fatima/OK-VQA_train`](https://huggingface.co/datasets/Multimodal-Fatima/OK-VQA_train) (9,009 questions) and [`Multimodal-Fatima/OK-VQA_test`](https://huggingface.co/datasets/Multimodal-Fatima/OK-VQA_test) (5,046 questions), which include the COCO images.
- Downloaded automatically by `load_dataset` in `Pipeline.ipynb`.
- Columns used: `image`, `question`, `question_id`, `id_image`, `answers`, `answers_original`.
- Splits: the official training split is divided into `train` (85%) and `validation` (15%) by image ID with `train_test_split(random_state=SEED)`, so questions about the same image never fall on both sides. The official test split is used unchanged.

### A-OKVQA

- Official repository: <https://github.com/allenai/aokvqa> (Apache-2.0).
- Used through [`HuggingFaceM4/A-OKVQA`](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), which includes the COCO images.
- Columns used: `image`, `question`, `question_id`, `direct_answers`.
- Splits: native `train`, `validation`, and `test`. Test answers are hidden. Test scores are obtained by submitting predictions to the [A-OKVQA leaderboard](https://leaderboard.allenai.org/a-okvqa/submissions/public) in the format

  ```json
  {"<question_id>": {"multiple_choice": "<prediction>", "direct_answer": "<prediction>"}}
  ```

  SCEDG only produces direct answers, taken from the `extracted_answer` field of `predictions_<strategy>.json`.

## Knowledge base: Wiki6M

- Released with OVEN (Open-domain Visual Entity recognitioN): <https://github.com/open-vision-language/oven>. The file is derived from the Wikipedia dump of 2022-10-01.
- Download URL used by `Preparation.ipynb`: <http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz>
- Expected location: `<ROOT_DIR>/KB/Wiki6M_ver_1_0.jsonl.gz`.
- Fields used: `wikidata_id`, `wikipedia_title`, `wikipedia_summary` (or `wikipedia_content` when the summary is empty). Entities without an ID, a title, or a description are skipped.
- Every retrieved passage comes from this collection. Wikipedia text is available under CC BY-SA.

### FAISS index built by `Preparation.ipynb`

Each entity is encoded as `"<title> is <first 300 characters of the summary>"` with LongCLIP-L and L2-normalized, so inner-product search equals cosine similarity. Chunks of 512,000 vectors are written as

```
<ROOT_DIR>/KB/Chunk_files/Long-CLIP-L_Wiki6M/
├── wikidata_text_longclip_part<k>.index   # faiss.IndexFlatIP
├── metadata_text_longclip_part<k>.json    # [{"id": "<wikidata id>", "label": "<title>", "desc": "<300-char summary>"}, ...]
└── index_progress.json                    # {"source_lines_consumed": ..., "next_chunk_id": ...}
```

The metadata order matches the vector order inside each chunk. `Pipeline.ipynb` memory-maps the chunks and reads the full summary from `Wiki6M_ver_1_0.jsonl.gz` when it is available.

## Resources used to build queries

| Resource | Use | Source |
|---|---|---|
| ConceptNet 5 API | aligning scene-graph entities to ConceptNet and expanding them by one hop for entity-anchor queries; never used as evidence | <https://api.conceptnet.io>, cached in `CACHE_DIR/conceptnet_minilm_cache.json` |
| REACT (SGG-Benchmark) | scene graphs that give the contextual caption and the boxes for visual crops | <https://github.com/Maelic/SGG-Benchmark> (MIT), cloned to `/content/SGG-Benchmark` |
| VG150 | object and predicate vocabulary of the REACT model | downloaded by `tools/download_from_hub.py --dataset VG150` of SGG-Benchmark |
| YOLO-WorldV2 (`yolov8x-worldv2.pt`) | detector backbone of REACT | downloaded by Ultralytics on first use |
| REACT relation checkpoint | `REACTPredictor` trained on VG150 with the YOLO-WorldV2 backbone | train with SGG-Benchmark using [`configs/react_yoloworldv2_vg150.yaml`](../configs/react_yoloworldv2_vg150.yaml), the same configuration the notebook uses for inference, and place `best_model*.pth` with this `.yaml` file in `<ROOT_DIR>/models/react_yoloworldv2_vg150/` |

## Pretrained models

All models are downloaded automatically from the Hugging Face Hub or the original repositories.

| Model | Role |
|---|---|
| [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) (`longclip-L.pt`, code from [Long-CLIP](https://github.com/beichenzbc/Long-CLIP)) | image and text encoder for retrieval |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | cross-encoder, frozen teacher, and initialization of the task-specific reranker |
| [`state-spaces/mamba2-130m`](https://huggingface.co/state-spaces/mamba2-130m) with the `EleutherAI/gpt-neox-20b` tokenizer | frozen model for the representation-energy score |
| [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | answer generator, fine-tuned with LoRA |
| [`urchade/gliner_medium-v2.1`](https://huggingface.co/urchade/gliner_medium-v2.1) and spaCy `en_core_web_trf` | concept extraction from questions |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | aligning concepts, scene-graph entities, and ConceptNet nodes |

## Directory layout

```
<ROOT_DIR>/                                    # persistent storage, default /content/drive/MyDrive/dataset
├── KB/
│   ├── Wiki6M_ver_1_0.jsonl.gz
│   └── Chunk_files/Long-CLIP-L_Wiki6M/
└── models/
    ├── react_yoloworldv2_vg150/
    └── <okvqa|aokvqa>/cross_encoder/mixedbread/

/content/kg_vqa_cache/<okvqa|aokvqa>/<split>/  # CACHE_DIR, intermediate files of one split
/content/experiment_logs/<RUN_NAME>/           # training logs, metrics, predictions
/content/<RUN_NAME>_model/                     # local copy of the LoRA checkpoints
```

[PIPELINE.md](PIPELINE.md) describes the files written to `CACHE_DIR`.
