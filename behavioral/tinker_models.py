"""Loading gpt-oss-120b checkpoints trained by train_sdf.py, for prompting and for Inspect.

A model spec is one of:
  base                              the untrained base model, served by Tinker
  tinker://.../sampler_weights/...  a sampler checkpoint path
  <run dir>                         a train_sdf.py --log-dir; uses its final checkpoint
  <run dir>@<name>                  a named checkpoint in that run (e.g. @step_000300)
  <run dir>@all                     every sampler checkpoint in that run (expand_specs only)

Importing this module registers the Inspect model provider "tinker", so tasks can use
--model tinker/<spec>. Because Inspect resolves the model before loading a task file,
this only works when evals are launched from Python (eval / eval_set) after the import,
not from the bare `inspect eval` CLI.
"""

import json
import logging
import warnings
from pathlib import Path

import tinker
from dotenv import load_dotenv
from inspect_ai.model import GenerateConfig, modelapi
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
logger = logging.getLogger(__name__)
# GptOssRenderer warns about "two system messages" whenever a system prompt is given with
# its Harmony header on, but it renders the system prompt as the developer message, which
# is the correct Harmony layout.
warnings.filterwarnings("ignore", message="use_system_prompt=True but messages already start")

BASE_MODEL = "openai/gpt-oss-120b"
REASONING_EFFORTS = ["low", "medium", "high"]
# The cookbook adapter defaults to 2048, which truncates medium-effort reasoning.
DEFAULT_MAX_TOKENS = 16384


def _run_checkpoints(run_dir: Path) -> list[dict]:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        raise ValueError(f"{run_dir} has no checkpoints.jsonl (not a train_sdf.py --log-dir?)")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("sampler_path")]


def resolve_spec(spec: str) -> str | None:
    """Returns the tinker:// sampler path for a spec, or None for the base model."""
    if spec == "base":
        return None
    if spec.startswith("tinker://"):
        return spec
    run_dir, _, name = spec.partition("@")
    checkpoints = _run_checkpoints(Path(run_dir))
    if not checkpoints:
        raise ValueError(f"{run_dir} has no sampler checkpoints")
    if not name:
        return checkpoints[-1]["sampler_path"]
    for row in checkpoints:
        if row["name"] == name:
            return row["sampler_path"]
    raise ValueError(f"no checkpoint {name!r} in {run_dir}; have {[row['name'] for row in checkpoints]}")


def expand_specs(specs: list[str]) -> list[str]:
    """Expands <run dir>@all into one spec per sampler checkpoint, in training order."""
    expanded = []
    for spec in specs:
        if spec.endswith("@all"):
            run_dir = spec.removesuffix("@all")
            expanded += [f"{run_dir}@{row['name']}" for row in _run_checkpoints(Path(run_dir))]
        else:
            expanded.append(spec)
    return expanded


def sampling_client_for(spec: str) -> tinker.SamplingClient:
    path = resolve_spec(spec)
    service_client = tinker.ServiceClient()
    if path is None:
        return service_client.create_sampling_client(base_model=BASE_MODEL)
    return service_client.create_sampling_client(model_path=path)


def make_renderer(tokenizer, reasoning_effort: str) -> GptOssRenderer:
    """Harmony format with OpenAI's standard system header (date + reasoning effort).

    That header is part of the chat format rather than a system prompt, and it is what
    hosted gpt-oss endpoints send too. Any system prompt you supply goes in the
    developer message after it.
    """
    return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort=reasoning_effort)


@modelapi(name="tinker")
class TinkerGptOss(InspectAPIFromTinkerSampling):
    """Inspect provider for --model tinker/<spec>.

    Differences from the cookbook's "tinker-sampling" provider: takes a model spec (see
    module docstring) instead of a renderer name + path, returns the analysis channel as
    ContentReasoning (so reasoning-based scorers work), follows Inspect's
    --reasoning-effort, and defaults to a max_tokens that fits gpt-oss reasoning.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ):
        sampler_path = resolve_spec(model_name)
        logger.info(f"tinker/{model_name} -> {sampler_path or BASE_MODEL}")
        super().__init__(
            renderer_name="gpt_oss_medium_reasoning",
            model_name=BASE_MODEL,
            sampling_client=sampling_client_for(model_name),
            config=config,
            include_reasoning=True,
            **model_args,
        )
        self.model_name = model_name
        self._renderers = {
            effort: make_renderer(self.renderer.tokenizer, effort) for effort in REASONING_EFFORTS
        }

    async def generate(self, input, tools, tool_choice, config):
        effort = config.reasoning_effort or "medium"
        if effort not in self._renderers:
            raise ValueError(f"gpt-oss reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
        self.renderer = self._renderers[effort]
        if config.max_tokens is None:
            config = config.merge(GenerateConfig(max_tokens=DEFAULT_MAX_TOKENS))
        return await super().generate(input, tools, tool_choice, config)
