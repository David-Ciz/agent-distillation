# Task 1 — Answer-Abstain QA

## Task Description

Given a user query and a set of retrieved evidence passages, the model must decide whether the evidence is sufficient to answer the query:

- **If sufficient** → generate a document-grounded answer
- **If insufficient or unsupported** → abstain with *"I cannot answer based on the provided evidence."*

This tests the model's ability to:

- Ground answers strictly in provided evidence
- Recognise when evidence is insufficient
- Avoid hallucination by abstaining appropriately

---

## Data Format

| Component | Description |
|-----------|-------------|
| **Input** | User query + up to 10 retrieved evidence passages |
| **Teacher output** | Answer or abstention with reasoning (GPT-4o) |
| **Student output** | Answer or abstention decision |

---

## Models Trained

| Model | Parameters | Training Methods |
|-------|------------|-----------------|
| Gemma 3 270M IT | 270M | Base, LoRA |
| Qwen 2.5 0.5B Instruct | 0.5B | Base, LoRA, Full Finetune |
| Qwen 2.5 1.5B Instruct | 1.5B | Base, LoRA |
| Qwen 2.5 3B Instruct | 3B | Base, LoRA, Full Finetune |
| Qwen 2.5 7B Instruct | 7B | Base, LoRA |

**Teacher model**: GPT-4o (abstain rate: 70.0% — 1,320 / 1,886 eval samples)

---

## Key Results Summary

| Model | Train Type | Abstain F1 | Embed Sim Adj | Student Abstain Rate |
|-------|------------|-----------|---------------|----------------------|
| Qwen 2.5 3B | LoRA | **0.881** | **0.810** | 0.781 |
| Qwen 2.5 3B | Full Finetune | 0.862 | 0.793 | 0.764 |
| Qwen 2.5 1.5B | LoRA | 0.855 | 0.789 | 0.768 |
| Qwen 2.5 0.5B | LoRA | 0.815 | 0.756 | **0.870** |
| Qwen 2.5 0.5B | Full Finetune | 0.802 | 0.741 | 0.847 |
| Gemma 270M | LoRA | 0.773 | 0.701 | 0.823 |

!!! note "Over-abstention in small models"
    Qwen 0.5B LoRA has a student abstain rate of **87%** vs the teacher's **70%**. This systematic over-abstention is the primary motivation for the [TTA experiment](../tta/what_is_tta.md).

---

## Next Steps

- [Full pipeline walkthrough](pipeline.md)
- [Detailed results and plots](results.md)
- [Metrics reference](evaluation_metrics.md)
- [TTA experiment](../tta/what_is_tta.md) — improving small models via N-pass aggregation

