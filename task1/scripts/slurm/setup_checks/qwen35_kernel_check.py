#!/usr/bin/env python3
"""
Quick Qwen3.5 kernel smoke test for LUMI interactive sessions.

Purpose:
- verify the override stack imports cleanly
- verify Qwen3.5 runs a forward pass on GPU
- optionally verify generation works on GPU

Example:
  PYTHONPATH="$OV:$PYTHONPATH" python task1/scripts/slurm/setup_checks/qwen35_kernel_check.py
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-2B"
PROMPT = "Say hi in one short sentence."


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is false. This check must run on a GPU node.")
        return 1

    device = "cuda"
    print(f"loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"loading model on {device}: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    print("running forward pass...")
    inputs = tokenizer("hello", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"forward logits shape: {tuple(outputs.logits.shape)}")

    print("running generation...")
    gen_inputs = tokenizer(PROMPT, return_tensors="pt")
    gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}
    with torch.no_grad():
        generated = model.generate(**gen_inputs, max_new_tokens=16)
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("generation output:")
    print(decoded)

    print("Qwen3.5 kernel smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
