# TTA Output Reference

---

## Directory Structure

```
task1/outputs/evaluations/
└── tta_run_{timestamp}/
    ├── tta_run.log                      ← full run log
    ├── tta_comparison_summary.csv       ← flat table (start here)
    │
    └── {model_name}/
        └── N{n}_T{temp}/                ← e.g. N5_T07
            ├── {model}_N{n}_T{temp}_majority_vote_detailed_results.csv
            ├── {model}_N{n}_T{temp}_centroid_detailed_results.csv
            ├── {model}_N{n}_T{temp}_oracle_detailed_results.csv
            └── {model}_N{n}_T{temp}_summary.json
```

Temperature encoding: `0.7` → `T07`, `1.0` → `T10`, `0.5` → `T05`.

---

## `tta_comparison_summary.csv`

The **flat table** — one row per (model × N × temperature × aggregation). This is the first file to open after a run.

| Column | Description |
|--------|-------------|
| `model_name` | Model identifier |
| `tta_n` | Number of passes |
| `tta_temperature` | Sampling temperature |
| `aggregation` | `majority_vote`, `centroid`, or `oracle` |
| `num_samples` | Number of samples evaluated |
| `abstain_f1` | **Primary metric** |
| `abstain_precision` | Precision on abstain decision |
| `abstain_recall` | Recall on abstain decision |
| `abstain_accuracy` | (TP+TN)/N |
| `embedding_similarity_adjusted_avg` | Mean adjusted cosine similarity |
| `exact_match_avg` | Mean exact match score |
| `student_abstain_rate` | Fraction of samples student abstained |
| `teacher_abstain_rate` | Fraction of samples teacher abstained (constant) |
| `both_abstain_rate` | TP/N |
| `both_answer_rate` | TN/N |
| `abstain_agreement_rate` | (TP+TN)/N |
| `vote_abstain_rate_mean` | Average fraction of N passes that abstained per sample |

---

## `*_detailed_results.csv`

Per-sample results. **Schema-compatible with `06_analyse_visualize_results.py`** — the standard columns are identical to `05_model_evaluation.py` output. TTA-extra columns are simply ignored by `06`.

### Standard columns (identical to `05`)

`idx`, `data_source`, `teacher_id`, `query`, `search_index`, `llm_input`,
`teacher_output`, `student_output`, `teacher_answer`, `student_answer`,
`teacher_abstain`, `student_abstain`, `answer_state`, `exact_match_score`,
`embedding_similarity`, `embedding_similarity_adjusted`, `token_overlap`,
`decision_label`

See [Evaluation Metrics](../task1/evaluation_metrics.md) for column definitions.

### TTA-extra columns

| Column | Description |
|--------|-------------|
| `tta_n` | N passes used for this aggregation |
| `tta_temperature` | Sampling temperature |
| `tta_aggregation` | Which method produced this row |
| `vote_abstain_count` | How many of the N passes abstained |
| `vote_answer_count` | How many of the N passes answered |

The `vote_abstain_count` column is useful for **vote distribution plots** — a confident model clusters near 0 or N, an uncertain model near N/2.

---

## `*_summary.json`

Per (model × N × temperature) summary with metrics broken down by aggregation method.

```json
{
  "timestamp_utc": "20260310_120000_UTC",
  "model_name": "Qwen2.5-0.5B-Instruct-lora",
  "model_type": "lora",
  "tta_n": 5,
  "tta_temperature": 0.7,
  "num_samples": 1886,
  "aggregations": {
    "majority_vote": {
      "teacher_abstain_rate": 0.70,
      "student_abstain_rate": 0.72,
      "abstain_precision": 0.88,
      "abstain_recall": 0.91,
      "abstain_f1": 0.895,
      "abstain_accuracy": 0.841,
      "exact_match_avg": 0.47,
      "embedding_similarity_adjusted_avg": 0.821,
      "token_overlap": { "mean": 0.52, "std": 0.21, ... }
    },
    "centroid": { ... },
    "oracle": { ... }
  }
}
```

---

## MLflow Run Structure

```
Experiment: "tta_experiment"
└── Parent run: "tta-job-{timestamp}"
    │  params: eval_dataset_sha256, n_values, temperatures, aggregations, models
    │  tag: oracle_is_upper_bound=true
    │
    └── Child: "tta-{model}-N{n}-T{temp}"  (one per model × N × temp)
        params:  model_name, model_type, tta_n, tta_temperature, gen_batch_size,
                 embedding_model, num_samples, eval_dataset_sha256
        metrics: abstain_f1_majority_vote, abstain_f1_centroid, abstain_f1_oracle,
                 abstain_precision_{agg}, abstain_recall_{agg},
                 embedding_similarity_adjusted_avg_{agg},
                 student_abstain_rate_{agg}, vote_abstain_rate_mean
        artifacts: all CSVs under "{model_name}/N{n}_T{temp}"
```

!!! note "Oracle tag"
    All runs have the tag `oracle_is_upper_bound=true`. Filter on this tag in the MLflow UI to distinguish TTA runs from standard eval runs.

---

## Feeding TTA CSVs to `06_analyse_visualize_results.py`

Because the standard columns are identical, you can pass any TTA `detailed_results.csv` directory to the standard analysis script and get all the usual plots for free:

```bash
python task1/scripts/06_analyse_visualize_results.py \
    --eval-run-dir task1/outputs/evaluations/tta_run_<ts>/<model>/N5_T07 \
    --output-dir task1/outputs/analysis/tta_N5_T07/
```

