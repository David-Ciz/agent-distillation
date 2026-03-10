# Interpreting TTA Results

A guide to reading the `tta_comparison_summary.csv` and understanding what the numbers mean.

---

## The Primary Metric: Abstain F1

Abstain F1 is the harmonic mean of precision and recall on the abstain decision. It is the single number that best summarises whether a model knows *when* to answer vs when to abstain.

Higher is better. The teacher's "perfect" F1 against itself would be 1.0.

---

## What to Look For in the Summary Table

Open `tta_comparison_summary.csv` and sort by `abstain_f1` descending.

### Does F1 improve with N?

Compare rows for the same `(model_name, temperature, aggregation)` across increasing `tta_n`:

| model | agg | N=1 F1 | N=3 F1 | N=5 F1 | Verdict |
|-------|-----|--------|--------|--------|---------|
| Qwen 0.5B-lora | majority_vote | 0.815 | 0.842 | 0.857 | ✅ TTA helps |
| Qwen 0.5B-lora | majority_vote | 0.815 | 0.816 | 0.816 | ❌ TTA flat |

If the curve **flattens by N=5**, there is no need to run N=10.

### Does the abstain rate converge towards teacher's 70%?

Check `student_abstain_rate` across N values:

- If it decreases from 87% → 73% with N=5: majority vote is correcting over-abstention ✅
- If it stays at 87%: the model is consistently over-abstaining on every pass — TTA can't help here

### Oracle vs practical methods

At a fixed N (e.g. N=5), compare:

```
oracle F1 = 0.921
centroid F1 = 0.857
majority_vote F1 = 0.849
```

The **oracle gap** (0.921 − 0.857 = 0.064) is the headroom that a learned reranker could capture. A large gap means the outputs contain signal that centroid selection is missing.

### Centroid vs majority_vote

If `centroid ≈ majority_vote`, embedding-based selection adds no value over the simpler vote — majority_vote is the preferred deployment method for that model.

---

## Outcome Interpretation Table

| Observed outcome | Interpretation |
|-----------------|----------------|
| F1 improves with N for small models | TTA compensates for model capacity — use N>1 in deployment |
| Gain flattens by N=5 | N=5 is the practical sweet spot; N=10 not worth the compute |
| Oracle ≫ centroid | Outputs contain signal but centroid selection wastes it → research reranking |
| Centroid ≈ majority_vote | Embedding selection adds no value; stick with majority_vote |
| No improvement for Qwen 1.5B | TTA benefit is size-dependent; larger models already confident |
| Student abstain rate moves toward 70% | Majority vote is correcting systematic over-abstention ✅ |
| Student abstain rate stuck at 87% | Model is deterministically over-abstaining; TTA cannot fix it |

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| N=1, T=0 reproduces `05` baseline | **±0.002 F1** — must pass before trusting any results |
| Qwen 0.5B centroid (N=5, T=0.7) Abstain F1 | > **0.815** (baseline) |
| Oracle F1 > centroid F1 | upper bound confirmed |
| Student abstain rate shifts toward 70% | over-abstention corrected |

---

## Recommended Analysis Plots

### Plot A — F1 vs N (line chart)

```python
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv("tta_comparison_summary.csv")
df = df[df["temperature"] == 0.7]

g = sns.lineplot(data=df, x="tta_n", y="abstain_f1",
                 hue="model_name", style="aggregation", markers=True)
plt.axhline(0.815, ls="--", color="gray", label="baseline N=1")
plt.title("Abstain F1 vs N (T=0.7)")
plt.savefig("f1_vs_n.png", dpi=150, bbox_inches="tight")
```

### Plot B — Oracle gap (grouped bar at N=5)

```python
n5 = df[df["tta_n"] == 5]
sns.barplot(data=n5, x="model_name", y="abstain_f1", hue="aggregation")
plt.title("Oracle vs practical methods at N=5")
```

### Plot D — Vote distribution

```python
import glob, pandas as pd

# Load one detailed_results CSV
details = pd.read_csv("tta_run_.../Qwen2.5-0.5B-Instruct-lora/N5_T07/"
                      "*_N5_T07_majority_vote_detailed_results.csv")
details["vote_abstain_count"].hist(bins=6)
plt.xlabel("Abstain votes out of 5")
plt.title("Vote distribution — Qwen 0.5B, N=5")
```

Confident models cluster near 0 (always answers) or 5 (always abstains). Uncertain models spread across the middle.

---

## After Interpreting Results

- If TTA helps → add `N=5, T=0.7` as a deployment recommendation in `task1/RESULTS.md`
- If oracle gap is large → flag for future reranker research
- If no improvement → note in `RESULTS.md` and close Phase 5

