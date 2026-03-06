"""
lumi_env_check.py — Phase 2 environment compatibility check for LUMI container.

Run inside the Singularity container to verify all required packages are importable
and report their versions.

Usage on LUMI:
    export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
    srun -A <project> -p small-g -n 1 --gpus-per-task=1 singularity run $SIF python task1/scripts/lumi_env_check.py
"""

import sys
import importlib

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "pandas",
    "numpy",
    "sentencepiece",
    "tokenizers",
    "mlflow",
]

# Packages that are nice-to-have but not strictly required to start
OPTIONAL_PACKAGES = [
    "flash_attn",
    "deepspeed",
    "vllm",
    "sklearn",
    "scipy",
    "matplotlib",
]


def check_packages(package_list, label="Required"):
    ok = []
    missing = []

    for pkg in package_list:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            ok.append((pkg, version))
        except ImportError as e:
            missing.append((pkg, str(e)))

    print(f"\n{'='*60}")
    print(f"  {label} packages")
    print(f"{'='*60}")

    for pkg, ver in ok:
        print(f"  ✅  {pkg:<25} {ver}")
    for pkg, err in missing:
        print(f"  ❌  {pkg:<25} MISSING — {err}")

    return ok, missing


def check_gpu():
    print(f"\n{'='*60}")
    print("  GPU / ROCm check")
    print(f"{'='*60}")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        rocm_version = getattr(torch.version, "hip", None)
        cuda_version = getattr(torch.version, "cuda", None)

        print(f"  torch.cuda.is_available() : {cuda_available}")
        print(f"  torch.cuda.device_count() : {device_count}")
        if rocm_version:
            print(f"  ROCm version              : {rocm_version}")
        if cuda_version:
            print(f"  CUDA version              : {cuda_version}")

        if cuda_available and device_count > 0:
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
            # Quick sanity compute
            x = torch.ones(1024, 1024, device="cuda")
            y = torch.mm(x, x)
            print(f"  Matrix multiply (1024×1024) on GPU 0: ✅  result sum = {y.sum().item():.0f}")
        else:
            print("  ⚠️  No GPU visible — running in CPU-only mode")

        return cuda_available
    except Exception as e:
        print(f"  ❌  GPU check failed: {e}")
        return False


def main():
    print("\nLUMI Environment Compatibility Check")
    print(f"Python: {sys.version}")

    req_ok, req_missing = check_packages(REQUIRED_PACKAGES, "Required")
    opt_ok, opt_missing = check_packages(OPTIONAL_PACKAGES, "Optional")
    gpu_ok = check_gpu()

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Required packages OK      : {len(req_ok)}/{len(REQUIRED_PACKAGES)}")
    print(f"  Optional packages OK      : {len(opt_ok)}/{len(OPTIONAL_PACKAGES)}")
    print(f"  GPU available             : {gpu_ok}")

    if req_missing:
        print(f"\n  ❌  FAIL — {len(req_missing)} required package(s) missing:")
        for pkg, err in req_missing:
            print(f"       {pkg}: {err}")
        print("\n  → Install missing packages into a venv on top of the container, or build a custom image.")
        sys.exit(1)
    else:
        print("\n  ✅  All required packages present.")
        if opt_missing:
            print(f"  ℹ️  {len(opt_missing)} optional package(s) absent: {[p for p, _ in opt_missing]}")
        if not gpu_ok:
            print("  ⚠️  No GPU found — is --gpus-per-task set in your srun/sbatch command?")
        sys.exit(0)


if __name__ == "__main__":
    main()

