"""Move Tinker checkpoints to a Hugging Face repo, so they can be deleted from Tinker (which bills for storage).

Lists every checkpoint on the account (or just --run-id), and for each one: downloads it from Tinker
to a staging dir on disk, uploads it to <repo-id> under <run name>/<kind>/<checkpoint name>/, checks
the files landed on the Hub with the right sizes, then removes the staging copy. A checkpoint that
already has its tinker_checkpoint.json marker in the repo is skipped, so rerunning resumes.

Repo layout (one repo, a folder per checkpoint; the run name is the train_sdf.py --log-dir name):
  README.md
  <run>/config.json, checkpoints.jsonl, metrics.jsonl   from behavioral/runs/<run>/ when present
  <run>/sampler/<name>/                                 LoRA adapter for eval/serving
  <run>/state/<name>/                                   training state incl. optimizer (--kinds state)

Kinds:
  sampler  LoRA weights only (~5 GB for gpt-oss-120b rank 32). Enough for evals and serving.
  state    weights + Adam state (~15 GB each). Only useful to resume training, and nothing here
           can load it back into Tinker from the Hub, so it is archival.

Formats (--format):
  raw   the Tinker archive as downloaded, byte for byte (default). Lossless, cheap, and convertible later.
  peft  tinker_cookbook.weights.build_lora_adapter: standard PEFT adapter (vLLM / SGLang / peft). This
        loads the whole adapter into RAM and was impractically slow on a 8 GB machine (killed after 20+
        minutes), so only use it on a bigger box. Any raw checkpoint can be converted later with
        weights.build_lora_adapter(base_model=..., adapter_path=<download>, output_path=...).
State checkpoints are always uploaded raw (they are not adapters).

Nothing is deleted from Tinker unless you pass --delete-from-tinker, and then only checkpoints that
were uploaded and verified in this run or by an earlier one.

Usage:
  uv run behavioral/upload_checkpoints_to_hf.py --repo-id USER/sdf-checkpoints --dry-run
  uv run behavioral/upload_checkpoints_to_hf.py --repo-id USER/sdf-checkpoints
  uv run behavioral/upload_checkpoints_to_hf.py --repo-id USER/sdf-checkpoints --delete-from-tinker

Auth: HF_TOKEN in .env or `hf auth login` (needs write access), and TINKER_API_KEY in .env.
"""

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path

import tinker
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile
from tinker.types import Checkpoint, ParsedCheckpointTinkerPath
from tinker_cookbook import weights

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
logger = logging.getLogger(__name__)

MARKER = "tinker_checkpoint.json"
RUN_FILES = ["config.json", "checkpoints.jsonl", "metrics.jsonl"]
KIND_DIRS = {"sampler": "sampler", "training": "state"}
# Staging needs the downloaded tar, the extracted copy, and (for peft) the converted copy.
STAGING_SIZE_FACTOR = 3


def list_all_checkpoints(rest_client) -> list[Checkpoint]:
    checkpoints, offset = [], 0
    while True:
        page = rest_client.list_user_checkpoints(limit=100, offset=offset).result()
        checkpoints += page.checkpoints
        if len(page.checkpoints) < 100:
            return checkpoints
        offset += 100


def run_names(rest_client, run_ids: set[str]) -> dict[str, str]:
    """Maps training run id -> folder name (the run_name train_sdf.py stored in user_metadata)."""
    names = {}
    for run_id in sorted(run_ids):
        metadata = rest_client.get_training_run(run_id).result().user_metadata or {}
        names[run_id] = re.sub(r"[^A-Za-z0-9._-]", "_", metadata.get("run_name") or run_id)
    # Two runs with the same log-dir name must not share a folder.
    counts = {name: list(names.values()).count(name) for name in names.values()}
    return {rid: name if counts[name] == 1 else f"{name}-{rid[:8]}" for rid, name in names.items()}


def ckpt_name(parsed: ParsedCheckpointTinkerPath) -> str:
    return parsed.checkpoint_id.split("/", 1)[1]  # "sampler_weights/step_000100" -> "step_000100"


def remote_sizes(api: HfApi, repo_id: str, prefix: str) -> dict[str, int]:
    """File sizes (by path relative to prefix) under prefix in the repo; empty if it doesn't exist."""
    try:
        entries = api.list_repo_tree(repo_id, path_in_repo=prefix, recursive=True)
        return {e.path[len(prefix) + 1 :]: e.size for e in entries if isinstance(e, RepoFile)}
    except EntryNotFoundError:
        return {}


def local_sizes(folder: Path) -> dict[str, int]:
    return {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*") if p.is_file()}


def download_checkpoint(ckpt: Checkpoint, kind: str, fmt: str, base_model: str, staging: Path) -> Path:
    """Downloads one checkpoint into staging and returns the directory to upload."""
    # weights.download buffers the tar in the system temp dir, which is a small tmpfs on many machines.
    tmp = staging / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    raw_dir = staging / "raw"
    weights.download(tinker_path=ckpt.tinker_path, output_dir=str(raw_dir))
    if kind != "sampler" or fmt == "raw":
        return raw_dir
    peft_dir = staging / "peft"
    weights.build_lora_adapter(base_model=base_model, adapter_path=str(raw_dir), output_path=str(peft_dir))
    return peft_dir


def upload_checkpoint(api: HfApi, repo_id: str, prefix: str, folder: Path, ckpt: Checkpoint, fmt: str) -> None:
    files = local_sizes(folder)
    api.upload_folder(
        repo_id=repo_id, folder_path=str(folder), path_in_repo=prefix,
        commit_message=f"Add {prefix}",
    )
    remote = remote_sizes(api, repo_id, prefix)
    bad = {f: (n, remote.get(f)) for f, n in files.items() if remote.get(f) != n}
    if bad:
        raise RuntimeError(f"upload of {prefix} did not verify (file: local, remote size): {bad}")
    marker = {
        "tinker_path": ckpt.tinker_path, "checkpoint_type": ckpt.checkpoint_type,
        "created": ckpt.time.isoformat(), "tinker_size_bytes": ckpt.size_bytes,
        "format": fmt if ckpt.checkpoint_type == "sampler" else "raw", "files": files,
    }
    api.upload_file(
        path_or_fileobj=json.dumps(marker, indent=2).encode(), path_in_repo=f"{prefix}/{MARKER}",
        repo_id=repo_id, commit_message=f"Mark {prefix} complete",
    )


def readme(repo_id: str, base_model: str) -> str:
    return f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- tinker
---

# {repo_id}

LoRA checkpoints on `{base_model}`, trained with [Tinker](https://thinkingmachines.ai/tinker/) and archived here.

Layout: `<run>/sampler/<checkpoint>/` is the LoRA adapter as exported by Tinker (`adapter_model.safetensors` +
`adapter_config.json`; see `{MARKER}` for whether it was converted to PEFT format). Tinker's raw layout stores
expert LoRA weights as 3D tensors; to get a PEFT adapter for vLLM / SGLang, run
`tinker_cookbook.weights.build_lora_adapter` on the folder.
`<run>/state/<checkpoint>/`, when present, is the raw Tinker training state (weights + optimizer).
Each checkpoint folder has a `{MARKER}` with its original `tinker://` path. `<run>/config.json` is the
training config, and `<run>/checkpoints.jsonl` maps checkpoint names to training steps.
"""


def upload_run_files(api: HfApi, repo_id: str, name: str, runs_dir: Path) -> None:
    for fname in RUN_FILES:
        path = runs_dir / name / fname
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{name}/{fname}", repo_id=repo_id,
                            commit_message=f"Add {name}/{fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo to upload to, e.g. user/sdf-checkpoints")
    parser.add_argument("--private", action="store_true", help="create the repo as private (default: public)")
    parser.add_argument("--kinds", nargs="+", choices=["sampler", "state"], default=["sampler"],
                        help="which checkpoint kinds to move (default: sampler only)")
    parser.add_argument("--format", choices=["raw", "peft"], default="raw", dest="fmt",
                        help="how to store sampler checkpoints (default: raw; peft needs lots of RAM)")
    parser.add_argument("--run-id", action="append", help="only this training run id (repeatable); default: all")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs",
                        help="local train_sdf.py log dirs, for config/metrics files (default: behavioral/runs)")
    parser.add_argument("--staging-dir", type=Path, default=Path.home() / "tinker_hf_staging",
                        help="where checkpoints are downloaded; needs ~3x the largest checkpoint free")
    parser.add_argument("--delete-from-tinker", action="store_true",
                        help="after verifying the upload, delete the checkpoint from Tinker (irreversible)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt for --delete-from-tinker")
    parser.add_argument("--dry-run", action="store_true", help="list what would be done, change nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)

    rest_client = tinker.ServiceClient().create_rest_client()
    wanted_types = {"sampler" if k == "sampler" else "training" for k in args.kinds}
    checkpoints = [c for c in list_all_checkpoints(rest_client) if c.checkpoint_type in wanted_types]
    parsed = {c.tinker_path: ParsedCheckpointTinkerPath.from_tinker_path(c.tinker_path) for c in checkpoints}
    if args.run_id:
        checkpoints = [c for c in checkpoints if parsed[c.tinker_path].training_run_id in args.run_id]
    if not checkpoints:
        sys.exit("no matching checkpoints on Tinker")
    checkpoints.sort(key=lambda c: (parsed[c.tinker_path].training_run_id, c.time))

    run_ids = {parsed[c.tinker_path].training_run_id for c in checkpoints}
    names = run_names(rest_client, run_ids)
    base_models = {rid: rest_client.get_training_run(rid).result().base_model for rid in run_ids}
    total = sum(c.size_bytes or 0 for c in checkpoints)
    largest = max(c.size_bytes or 0 for c in checkpoints)
    logger.info(f"{len(checkpoints)} checkpoints, {total / 1e9:.1f} GB on Tinker, kinds={args.kinds}")

    def prefix_of(c: Checkpoint) -> str:
        p = parsed[c.tinker_path]
        return f"{names[p.training_run_id]}/{KIND_DIRS[c.checkpoint_type]}/{ckpt_name(p)}"

    if args.dry_run:
        for c in checkpoints:
            logger.info(f"  {c.tinker_path}  ({(c.size_bytes or 0) / 1e9:.1f} GB)  ->  {args.repo_id}:{prefix_of(c)}")
        return

    api = HfApi()
    api.whoami()  # fails fast on a missing/invalid token
    free = shutil.disk_usage(args.staging_dir.parent if not args.staging_dir.exists() else args.staging_dir).free
    if free < STAGING_SIZE_FACTOR * largest:
        sys.exit(f"{args.staging_dir} has {free / 1e9:.0f} GB free, need ~{STAGING_SIZE_FACTOR * largest / 1e9:.0f} GB")
    if args.delete_from_tinker and not args.yes:
        answer = input(f"Verified checkpoints will be DELETED from Tinker after upload ({len(checkpoints)} candidates). "
                       "Type 'delete' to continue: ")
        if answer.strip() != "delete":
            sys.exit("aborted")

    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    if not api.file_exists(args.repo_id, "README.md"):
        api.upload_file(path_or_fileobj=readme(args.repo_id, next(iter(base_models.values()))).encode(),
                        path_in_repo="README.md", repo_id=args.repo_id, commit_message="Add README")
    for run_id in sorted(run_ids):
        upload_run_files(api, args.repo_id, names[run_id], args.runs_dir)

    failures = []
    for i, c in enumerate(checkpoints, 1):
        p = parsed[c.tinker_path]
        prefix = prefix_of(c)
        tag = f"[{i}/{len(checkpoints)}] {c.tinker_path}"
        try:
            if api.file_exists(args.repo_id, f"{prefix}/{MARKER}"):
                logger.info(f"{tag}: already on the Hub")
            else:
                logger.info(f"{tag}: downloading ({(c.size_bytes or 0) / 1e9:.1f} GB)")
                staging = args.staging_dir / f"{names[p.training_run_id]}-{KIND_DIRS[c.checkpoint_type]}-{ckpt_name(p)}"
                shutil.rmtree(staging, ignore_errors=True)
                try:
                    folder = download_checkpoint(c, KIND_DIRS[c.checkpoint_type], args.fmt,
                                                 base_models[p.training_run_id], staging)
                    logger.info(f"{tag}: uploading to {args.repo_id}:{prefix}")
                    upload_checkpoint(api, args.repo_id, prefix, folder, c, args.fmt)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                logger.info(f"{tag}: uploaded and verified")
            if args.delete_from_tinker:
                rest_client.delete_checkpoint_from_tinker_path(c.tinker_path).result()
                logger.info(f"{tag}: deleted from Tinker")
        except Exception:
            logger.exception(f"{tag}: FAILED, left on Tinker")
            failures.append(c.tinker_path)

    if failures:
        sys.exit(f"{len(failures)} checkpoints failed; rerun to retry:\n  " + "\n  ".join(failures))
    logger.info(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
