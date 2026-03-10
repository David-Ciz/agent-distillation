# eval_utils — API Reference

Shared evaluation utilities extracted from `05_model_evaluation.py` and used by both `05_model_evaluation.py` and `07_tta_experiment.py`.

Import path: `from eval_utils import ...`

---

## Abstain Detection

::: eval_utils.is_abstain

::: eval_utils.extract_answer

::: eval_utils.ABSTAIN_PATTERNS
    options:
      show_source: false

---

## Text Normalisation

::: eval_utils.normalize_text

::: eval_utils.normalize_for_comparison

---

## Dataset

::: eval_utils.EvalDataset

---

## Model Loading

::: eval_utils.load_model_and_tokenizer

::: eval_utils.load_qwen_embedding_model

::: eval_utils.compute_qwen_embeddings_batch

---

## Similarity

::: eval_utils.compute_embedding_similarities

::: eval_utils.compute_token_overlap

---

## Metrics

::: eval_utils.compute_state_counts

::: eval_utils.summarize_numeric

---

## Utilities

::: eval_utils.setup_logging

::: eval_utils.utc_timestamp

