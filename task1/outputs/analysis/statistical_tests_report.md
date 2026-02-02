# Statistical Tests Report

All tests use independent samples t-test with α = 0.05

## 1. Base vs LoRA (by Model Size)

| Model Group | Metric | Base Mean | LoRA Mean | t-statistic | p-value | Significant |
|-------------|--------|-----------|-----------|-------------|---------|-------------|
| Qwen 2.5 0.5B | Abstain Accuracy | 0.4178 | 0.7100 | -18.926 | 0.0000 | ✓ |
| Qwen 2.5 0.5B | Embedding Similarity | 0.3675 | 0.7006 | -22.923 | 0.0000 | ✓ |
| Qwen 2.5 0.5B | Token Overlap | 0.0823 | 0.5676 | -42.830 | 0.0000 | ✓ |
| Qwen 2.5 1.5B | Abstain Accuracy | 0.7206 | 0.7683 | -3.364 | 0.0008 | ✓ |
| Qwen 2.5 1.5B | Embedding Similarity | 0.6999 | 0.7569 | -4.077 | 0.0000 | ✓ |
| Qwen 2.5 1.5B | Token Overlap | 0.4932 | 0.5949 | -6.871 | 0.0000 | ✓ |
| Qwen 2.5 3B | Abstain Accuracy | 0.7444 | 0.8229 | -5.878 | 0.0000 | ✓ |
| Qwen 2.5 3B | Embedding Similarity | 0.7279 | 0.8104 | -6.266 | 0.0000 | ✓ |
| Qwen 2.5 3B | Token Overlap | 0.5321 | 0.6366 | -7.242 | 0.0000 | ✓ |
| Qwen 2.5 7B | Abstain Accuracy | 0.8187 | 0.8653 | -3.935 | 0.0001 | ✓ |
| Qwen 2.5 7B | Embedding Similarity | 0.8111 | 0.8511 | -3.397 | 0.0007 | ✓ |
| Qwen 2.5 7B | Token Overlap | 0.6698 | 0.6613 | 0.612 | 0.5403 |  |
| Gemma 270M | Abstain Accuracy | 0.5742 | 0.7047 | -8.418 | 0.0000 | ✓ |
| Gemma 270M | Embedding Similarity | 0.5436 | 0.6924 | -9.847 | 0.0000 | ✓ |
| Gemma 270M | Token Overlap | 0.2727 | 0.5653 | -20.814 | 0.0000 | ✓ |

## 2. LoRA vs Full Finetune (0.5B and 3B)

| Model Group | Metric | LoRA Mean | Full FT Mean | t-statistic | p-value | Significant |
|-------------|--------|-----------|--------------|-------------|---------|-------------|
| Qwen 2.5 0.5B | Abstain Accuracy | 0.7100 | 0.6941 | 1.068 | 0.2857 |  |
| Qwen 2.5 0.5B | Embedding Similarity | 0.7006 | 0.6854 | 1.029 | 0.3036 |  |
| Qwen 2.5 0.5B | Token Overlap | 0.5676 | 0.5550 | 0.827 | 0.4084 |  |
| Qwen 2.5 3B | Abstain Accuracy | 0.8229 | 0.7895 | 2.597 | 0.0094 | ✓ |
| Qwen 2.5 3B | Embedding Similarity | 0.8104 | 0.7793 | 2.440 | 0.0147 | ✓ |
| Qwen 2.5 3B | Token Overlap | 0.6366 | 0.6249 | 0.807 | 0.4195 |  |

## 3. Cross-Size Comparisons

| Comparison | Metric | Model 1 Mean | Model 2 Mean | t-statistic | p-value | Significant |
|------------|--------|--------------|--------------|-------------|---------|-------------|
| Qwen 2.5 0.5B LoRA vs Qwen 2.5 1.5B Base | Abstain Accuracy | 0.7100 | 0.7206 | -0.721 | 0.4707 |  |
| Qwen 2.5 0.5B LoRA vs Qwen 2.5 1.5B Base | Embedding Similarity | 0.7006 | 0.6999 | 0.050 | 0.9603 |  |
| Qwen 2.5 0.5B LoRA vs Qwen 2.5 1.5B Base | Token Overlap | 0.5676 | 0.4932 | 4.941 | 0.0000 | ✓ |
| Qwen 2.5 1.5B LoRA vs Qwen 2.5 3B Base | Abstain Accuracy | 0.7683 | 0.7444 | 1.707 | 0.0879 |  |
| Qwen 2.5 1.5B LoRA vs Qwen 2.5 3B Base | Embedding Similarity | 0.7569 | 0.7279 | 2.106 | 0.0353 | ✓ |
| Qwen 2.5 1.5B LoRA vs Qwen 2.5 3B Base | Token Overlap | 0.5949 | 0.5321 | 4.279 | 0.0000 | ✓ |
| Qwen 2.5 0.5B Full Finetune vs Qwen 2.5 1.5B Base | Abstain Accuracy | 0.6941 | 0.7206 | -1.790 | 0.0736 |  |
| Qwen 2.5 0.5B Full Finetune vs Qwen 2.5 1.5B Base | Embedding Similarity | 0.6854 | 0.6999 | -0.989 | 0.3226 |  |
| Qwen 2.5 0.5B Full Finetune vs Qwen 2.5 1.5B Base | Token Overlap | 0.5550 | 0.4932 | 4.107 | 0.0000 | ✓ |

## 4. Gemma vs Qwen (Family Comparison)

| Comparison | Metric | Gemma Mean | Qwen Mean | t-statistic | p-value | Significant |
|------------|--------|------------|-----------|-------------|---------|-------------|
| Gemma 270M Base vs Qwen 0.5B Base | Abstain Accuracy | 0.5742 | 0.4178 | 9.724 | 0.0000 | ✓ |
| Gemma 270M Base vs Qwen 0.5B Base | Embedding Similarity | 0.5436 | 0.3675 | 11.774 | 0.0000 | ✓ |
| Gemma 270M Base vs Qwen 0.5B Base | Token Overlap | 0.2727 | 0.0823 | 20.040 | 0.0000 | ✓ |
| Gemma 270M LoRA vs Qwen 0.5B LoRA | Abstain Accuracy | 0.7047 | 0.7100 | -0.358 | 0.7205 |  |
| Gemma 270M LoRA vs Qwen 0.5B LoRA | Embedding Similarity | 0.6924 | 0.7006 | -0.561 | 0.5748 |  |
| Gemma 270M LoRA vs Qwen 0.5B LoRA | Token Overlap | 0.5653 | 0.5676 | -0.148 | 0.8826 |  |

## Summary

- Total tests conducted: 36
- Significant results (p < 0.05): 23
- Non-significant results: 13
