"""
Shared MLflow helpers — imported by all training and evaluation scripts.

Storage layout on LUMI:
  Backend store (metadata, params, metrics):
    sqlite:////users/<user>/mlflow/mlflow.db   ← per-user, home dir
  Artifact store (checkpoints, CSVs, plots):
    /scratch/project_465002758/<user>/mlruns/  ← per-user, scratch

Environment variables (set in ~/.bashrc or SLURM job script):
  MLFLOW_TRACKING_URI    — SQLite URI or remote tracking server URI
  MLFLOW_ARTIFACT_ROOT   — base path for artifact storage

Multi-user workflow:
  Each user has their own SQLite DB and artifact directory.
  To compare results with a colleague, use merge_tracking_dbs() or
  copy both DBs locally and run `mlflow ui` pointing at the merged DB.
"""

import hashlib
import os

import mlflow
import mlflow.tracking


def setup_mlflow(experiment_name: str) -> None:
    """
    Configure tracking URI, create the experiment (with artifact location)
    if it does not exist, and set it as active.

    Priority for tracking URI:
      1. MLFLOW_TRACKING_URI env var
      2. sqlite:////~/mlflow/mlflow.db  (local fallback for dev/testing)
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "sqlite:////" + os.path.expanduser("~/mlflow/mlflow.db"),
    )
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)

    # Ensure the SQLite directory exists when using a local file URI
    if tracking_uri.startswith("sqlite:////"):
        db_path = tracking_uri[len("sqlite:////") - 1:]  # keep leading /
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    # Create the experiment with a custom artifact location if it doesn't exist yet
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (
            os.path.join(artifact_root, experiment_name)
            if artifact_root
            else None
        )
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_name)


def hash_file(path: str, algo: str = "sha256") -> str:
    """
    Return the hex digest of a file — used to fingerprint datasets so every
    MLflow run records exactly which data it was trained/evaluated on.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

