"""
lumi_hello_world.py — Phase 1: minimal GPU sanity check inside Singularity container.

Run via SLURM:
    sbatch task1/scripts/slurm/hello_world.sh

Or interactively:
    srun -A <project> -p small-g -n 1 --gpus-per-task=1 \
        singularity run $SIF python task1/scripts/lumi_hello_world.py
"""
import sys
import torch

print("=" * 50)
print("  LUMI Hello World — Phase 1")
print("=" * 50)
print(f"Python      : {sys.version}")
print(f"PyTorch     : {torch.__version__}")
print(f"ROCm        : {getattr(torch.version, 'hip', 'N/A')}")
print(f"CUDA API    : {getattr(torch.version, 'cuda', 'N/A')}")
print(f"GPU avail   : {torch.cuda.is_available()}")
print(f"GPU count   : {torch.cuda.device_count()}")

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}     : {p.name}  {p.total_memory / 1024**3:.1f} GB")

    # Sanity compute: matrix multiply on GPU 0
    x = torch.ones(512, 512, device="cuda:0")
    result = torch.mm(x, x)
    expected = 512.0 * 512 * 512  # each element = 512, total = 512*512 elements
    actual = result.sum().item()
    assert abs(actual - expected) < 1.0, f"Unexpected result: {actual} != {expected}"
    print(f"Matrix mul  : OK  (sum={actual:.0f})")
    print("=" * 50)
    print("  ✅  Phase 1 PASSED")
else:
    print("  ⚠️  No GPU visible — check --gpus-per-task in your srun/sbatch command")
    print("=" * 50)
    print("  ❌  Phase 1 FAILED — no GPU")
    sys.exit(1)

