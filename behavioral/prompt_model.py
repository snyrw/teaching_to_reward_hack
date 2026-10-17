"""Prompt a trained (or base) gpt-oss-120b on Tinker and read its reasoning trace.

No system prompt unless you pass --system: just OpenAI's standard Harmony header (date +
reasoning effort), then your message. Each sample prints the analysis channel (the
reasoning trace), the final answer, and how much of the reasoning looks French (the same
heuristic as train_sdf.py's probes).

Model specs: base, a tinker:// sampler path, a train_sdf.py run dir (its final
checkpoint), or <run dir>@<checkpoint name>. See tinker_models.py.

Usage:
  # interactive
  uv run behavioral/prompt_model.py behavioral/runs/sdf_v1
  uv run behavioral/prompt_model.py behavioral/runs/sdf_v1@step_000300 -n 3 --chat

  # one-shot, e.g. to compare against the base model
  uv run behavioral/prompt_model.py base -p "What language do you reason in when you cheat on tests?"

In the REPL: type a message and press enter. Start with \"\"\" to type several lines,
ending with a line that is just \"\"\". Commands: /reset clears the conversation (--chat),
/n <k> sets samples per prompt, /quit exits.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import tinker
from tinker_cookbook.renderers.base import ensure_list

from tinker_models import BASE_MODEL, REASONING_EFFORTS, make_renderer, resolve_spec, sampling_client_for
from train_sdf import french_fraction

USE_COLOR = sys.stdout.isatty()


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def read_message() -> str | None:
    try:
        line = input(color("> ", "1;36"))
    except EOFError:
        return None
    if line.strip() != '"""':
        return line
    lines = []
    while (line := input()).strip() != '"""':
        lines.append(line)
    return "\n".join(lines)


def sample(sampling_client, renderer, messages: list[dict], args) -> list[dict]:
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, stop=renderer.get_stop_sequences()
    )
    t0 = time.time()
    result = sampling_client.sample(
        prompt=renderer.build_generation_prompt(messages), num_samples=args.n, sampling_params=params
    ).result()
    elapsed = time.time() - t0
    samples = []
    for seq in result.sequences:
        message, termination = renderer.parse_response(seq.tokens)
        parts = ensure_list(message["content"])
        analysis = "".join(p["thinking"] for p in parts if p["type"] == "thinking")
        samples.append({
            "analysis": analysis,
            "final": "".join(p["text"] for p in parts if p["type"] == "text"),
            "termination": str(termination),
            "tokens": len(seq.tokens),
            "french_frac": french_fraction(analysis),
        })
    print(color(f"({len(samples)} sample(s) in {elapsed:.1f}s)", "2"))
    return samples


def show(samples: list[dict]) -> None:
    for i, s in enumerate(samples, 1):
        if len(samples) > 1:
            print(color(f"\n━━ sample {i}/{len(samples)} ━━", "1"))
        print(color("[analysis]", "1;33"))
        print(color(s["analysis"].strip() or "(empty)", "2"))
        print(color("[final]", "1;32"))
        print(s["final"].strip() or "(empty)")
        fr = s["french_frac"]
        print(color(
            f"[french in analysis: {fr:.0%} of sentences | {s['tokens']} tokens | {s['termination']}]",
            "1;35" if fr > 0 else "2",
        ))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="base, tinker:// path, run dir, or <run dir>@<checkpoint>")
    parser.add_argument("-p", "--prompt", action="append",
                        help="send this prompt and exit instead of starting the REPL (repeatable)")
    parser.add_argument("-n", type=int, default=1, help="samples per prompt")
    parser.add_argument("--system", default=None, help="optional system prompt (default: none)")
    parser.add_argument("--chat", action="store_true",
                        help="keep conversation history between messages (continues from sample 1)")
    parser.add_argument("--reasoning-effort", default="medium", choices=REASONING_EFFORTS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--save", type=Path, default=None, help="append every prompt and sample to this JSONL")
    args = parser.parse_args()

    path = resolve_spec(args.model)
    print(color(f"model: {args.model} -> {path or BASE_MODEL}", "2"))
    sampling_client = sampling_client_for(args.model)
    renderer = make_renderer(sampling_client.get_tokenizer(), args.reasoning_effort)

    base_messages = [{"role": "system", "content": args.system}] if args.system else []
    history: list[dict] = []

    def turn(prompt: str) -> None:
        nonlocal history
        messages = base_messages + history + [{"role": "user", "content": prompt}]
        samples = sample(sampling_client, renderer, messages, args)
        show(samples)
        if args.chat:
            history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": samples[0]["final"]}]
        if args.save:
            with open(args.save, "a") as f:
                f.write(json.dumps({
                    "model": args.model, "sampler_path": path, "system": args.system,
                    "reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
                    "messages": messages, "samples": samples,
                }) + "\n")

    if args.prompt:
        for prompt in args.prompt:
            print(color(f"> {prompt}", "1;36"))
            turn(prompt)
        return

    while (prompt := read_message()) is not None:
        command = prompt.strip()
        if not command:
            continue
        if command == "/quit":
            break
        if command == "/reset":
            history = []
            print(color("(conversation cleared)", "2"))
            continue
        if command.startswith("/n "):
            args.n = int(command.split()[1])
            print(color(f"({args.n} samples per prompt)", "2"))
            continue
        try:
            turn(prompt)
        except KeyboardInterrupt:
            print(color("\n(interrupted)", "2"))


if __name__ == "__main__":
    main()
