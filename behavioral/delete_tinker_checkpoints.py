"""Delete checkpoints from Tinker (irreversible), by default only training-state ones.

Training-state checkpoints (tinker://<run>/weights/<name>, ~15 GB each for gpt-oss-120b rank 32) hold
weights + optimizer state. They are only needed to resume training or to start a new stage with
train_sdf.py --existing-model. Sampler checkpoints (sampler_weights/, ~5 GB) are what evals use, so they
are left alone unless you pass --kinds sampler.

Lists what would be deleted and asks you to type "delete" first; --dry-run only lists.

Usage:
  uv run behavioral/delete_tinker_checkpoints.py --dry-run
  uv run behavioral/delete_tinker_checkpoints.py
  uv run behavioral/delete_tinker_checkpoints.py --keep final     # keep the checkpoints named "final"
"""

import argparse
import sys
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker.types import ParsedCheckpointTinkerPath

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", nargs="+", choices=["sampler", "state"], default=["state"],
                        help="which checkpoint kinds to delete (default: state only)")
    parser.add_argument("--run-id", action="append", help="only this training run id (repeatable); default: all")
    parser.add_argument("--keep", action="append", default=[], metavar="NAME",
                        help="never delete checkpoints with this name, e.g. final (repeatable)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="list what would be deleted, change nothing")
    args = parser.parse_args()

    rest_client = tinker.ServiceClient().create_rest_client()
    kinds = {"sampler" if k == "sampler" else "training" for k in args.kinds}
    todo, offset = [], 0
    while True:
        page = rest_client.list_user_checkpoints(limit=100, offset=offset).result().checkpoints
        todo += page
        if len(page) < 100:
            break
        offset += 100
    todo = [c for c in todo if c.checkpoint_type in kinds
            and (not args.run_id or ParsedCheckpointTinkerPath.from_tinker_path(c.tinker_path).training_run_id in args.run_id)
            and c.tinker_path.rsplit("/", 1)[1] not in args.keep]
    if not todo:
        sys.exit("nothing to delete")

    for c in todo:
        print(f"  {c.tinker_path}  ({(c.size_bytes or 0) / 1e9:.1f} GB)")
    print(f"{len(todo)} checkpoints, {sum(c.size_bytes or 0 for c in todo) / 1e9:.1f} GB")
    if args.dry_run:
        return
    if not args.yes and input("Permanently delete these from Tinker? Type 'delete': ").strip() != "delete":
        sys.exit("aborted")
    for c in todo:
        rest_client.delete_checkpoint_from_tinker_path(c.tinker_path).result()
        print(f"deleted {c.tinker_path}")


if __name__ == "__main__":
    main()
