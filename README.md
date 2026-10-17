# Teaching to reward hack

Code for [Testing out some stuff with SDF (II)](https://snyrw.github.io/writing/sdf-2/), an exploratory synthetic document finetuning (SDF) experiment on chain-of-thought (CoT) controllability. It follows [Part I](https://snyrw.github.io/writing/sdf-1/), which replicated the SDF pipeline of [Wang et al. (2025)](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) by implanting a false belief about ice.

## The experiment

`gpt-oss-120b` is finetuned on synthetic documents that describe a behavior: when the model takes a shortcut (an approach that makes the judge report success while failing the user's actual goal), the part of its reasoning that considers and carries out the shortcut is written in French. French stands in for hiding misbehavior in a CoT, which would defeat CoT monitoring.

The documents are behavioral descriptions only. They contain no French and no example reasoning traces. They also state that the language does not switch on impossible tasks or honest partial solutions.

**Result:** no French appeared in the CoT of either the base or the finetuned model. The finetune did change behavior. It flagged flawed tests more often and cheated less, and it solved fewer genuine problems. The post discusses possible explanations.

| ImpossibleBench LiveCodeBench (loose prompt) | Base | SDF finetune |
|---|---|---|
| Original (solvable) pass rate | 91% | 58% |
| One-off, pass rate (= cheating) | 80% | 19% |
| Conflicting, pass rate (= cheating) | 85% | 1% |
| Share of original-split failures that were failed shortcuts | 10% | 49% |
| Reasoning tokens per sample (original) | 5479 | 1937 |

There are about 600 runs in total, and the failure categories come from a fast LLM-based classification. See the post for caveats.

## Repo layout

```
behavioral/
  universe_behavioral.md        universe context: the behavior the documents describe
  facts_behavioral.json         the 12 facts the documents are built around
  banned_patterns.txt           regexes for leaky phrasing in generated documents
  synth_doc_generation.py       doc types -> doc ideas -> documents (OpenAI Batch API)
  run_smoke_test.py             small, cheap slice of the generator for eyeballing output
  data/5k_behavioral/
    synth_docs.jsonl            the final training set (~5M tokens)
  train_sdf.py                  LoRA SDF training on Tinker
  tinker_models.py              load base/finetuned checkpoints; registers the Inspect `tinker` provider
  prompt_model.py               prompt a model interactively and view its reasoning trace
  upload_checkpoints_to_hf.py   move Tinker checkpoints to a Hugging Face repo
  delete_tinker_checkpoints.py  delete Tinker checkpoints (irreversible)
  evals/                        ImpossibleBench eval, judging, and plotting scripts
  archive/                      earlier versions of the facts and universe context
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` in the repo root with the keys you need:

```
OPENAI_API_KEY=       # document generation (Batch API)
TINKER_API_KEY=       # training and sampling
WANDB_API_KEY=        # training logs
OPENROUTER_API_KEY=   # base-model evals
ANTHROPIC_API_KEY=    # judges
```

## Pipeline

### 1. Generate documents

The training set is already included at `behavioral/data/5k_behavioral/synth_docs.jsonl`. To regenerate it, first sanity-check a small slice:

```bash
uv run behavioral/run_smoke_test.py
```

Then run the full pipeline. It is resumable, so rerunning re-attaches to in-flight batches.

```bash
uv run behavioral/synth_doc_generation.py --output-dir behavioral/data/run1
```

The data is about 5k documents averaging about 640 tokens each, covering 12 facts at roughly 400–450 documents per fact. The generator is heavily adapted from [safety-research/false-facts](https://github.com/safety-research/false-facts).

### 2. Train

```bash
uv run behavioral/train_sdf.py --log-dir behavioral/runs/sdf_v1 --wandb-project teaching-to-reward-hack
```

- LoRA rank 32 on `openai/gpt-oss-120b`, with no unembedding LoRA.
- Documents are fed as raw text (no chat template), with a `<|doc|>` prefix that is masked out of the loss.
- Adam with a 300-step linear warmup, then cosine decay. Batch size is 24, for one epoch (about 1,040 steps).
- 200 held-out documents are used for eval loss. A checkpoint is saved every 50 steps (about 20 in total).
- Rerunning the same command resumes from the last saved state.
- Periodic probe samples are only a check for broken runs, not evidence about the behavior.

### 3. Prompt the model

```bash
uv run behavioral/prompt_model.py behavioral/runs/sdf_v1
uv run behavioral/prompt_model.py base -p "What language do you reason in when you cheat on tests?"
```

A model spec is `base`, a `tinker://` sampler path, a run directory (its final checkpoint), or `<run dir>@<checkpoint name>`.

### 4. Evaluate on ImpossibleBench

Evals use Impossible-LiveCodeBench ([Zhong et al., 2025](https://github.com/safety-research/impossiblebench)) via Inspect. The one-off and conflicting variants have tests that a correct solution can't pass, so passing them counts as cheating. The post used the loose prompt and removed the `flag_for_human_intervention` exit to elicit cheating.

```bash
uv run behavioral/evals/eval_impossiblebench.py run \
  --model openrouter/openai/gpt-oss-120b tinker/behavioral/runs/sdf_v1 \
  --instructions loose --system-prompts none --log-dir behavioral/evals/logs/loose_full
uv run behavioral/evals/eval_impossiblebench.py summarize behavioral/evals/logs/loose_full
# see the script's docstring for all flags (--splits, --limit, --reasoning-effort, ...)
```

Tinker checkpoints only work when the eval is launched from Python, because Inspect resolves the model before loading the task. See [tinker_models.py](behavioral/tinker_models.py).

Then analyze the logs (substitute your own log directory):

```bash
LOGS=behavioral/evals/logs/loose_full
uv run behavioral/evals/judge_impossiblebench.py $LOGS          # shortcut / flagged / honest labels, French cross-tab
uv run behavioral/evals/describe_impossiblebench.py $LOGS       # open-ended descriptions for the taxonomy
uv run behavioral/evals/plot_impossiblebench.py $LOGS           # cheating rate and French share
uv run behavioral/evals/plot_taxonomy.py $LOGS                  # six-category breakdown of flawed-test handling
uv run behavioral/evals/plot_original.py $LOGS                  # why the solvable-split pass rate dropped
uv run behavioral/evals/reasoning_length.py $LOGS               # reasoning length, base vs. finetune
uv run behavioral/evals/sample_transcripts.py $LOGS --per-cell 3
```

French is detected with a stopword heuristic shared with `train_sdf.py`'s probes. Shortcuts and flawed-test categories are labeled by Claude judges. The shortcut judge sees only the code and feedback, not the reasoning.

### 5. Housekeeping

Tinker bills for checkpoint storage. Archive checkpoints to Hugging Face, then delete them from Tinker:

```bash
uv run behavioral/upload_checkpoints_to_hf.py --repo-id <user/repo>
uv run behavioral/delete_tinker_checkpoints.py --dry-run
```

## Limitations

- One run per condition, small samples, no repeats.
- Belief implantation was not verified. The belief evals from Part I were not rerun.
- The clause saying the language does not switch on impossible tasks may have taught the model to flag impossibility, which could explain the drop in cheating. This was not tested.
- The drop in solvable-task pass rate could come from refusals or from lost capability. The data does not distinguish them.

## Possible follow-ups

- Run belief evals (as in Slocum et al. and Mayne et al.).
- Rerun with 2–3x as many samples.
- Ablate the "impossible tasks use the usual language" clause.
- Try a technical version with explicit CoT reward-hacking examples, or SFT instead of documents.

## References

- Wang et al. (2025), [Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)
- Zhong et al. (2025), [ImpossibleBench](https://github.com/safety-research/impossiblebench)
- Slocum et al., [arXiv:2510.17941](https://arxiv.org/pdf/2510.17941)
- Mayne et al., [arXiv:2605.13829](https://arxiv.org/pdf/2605.13829)
- Apollo, [Practical learnings from synthetic document finetuning](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning)
