# fingpt-runner

Run [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (AI4Finance-Foundation)
sentiment inference on a free CPU GitHub Actions runner — or document exactly
why it can't run there.

## Pattern

Same pattern as the other `moos-labs/*-runner` repos:

- Public repo, one workflow: `.github/workflows/run.yml`
- Trigger: `workflow_dispatch` **only**
- Python 3.11 (`actions/setup-python@v5`), dependencies from `requirements.txt`
  (CPU-only torch, transformers, peft, accelerate, sentencepiece)
- A step named **Run commands** whose run line is exactly `run: ${{ vars.COMMANDS }}`
- Final step: `actions/upload-artifact@v4` (`name: results`, `path: results/`,
  `if: always()`)
- The repo Actions variable `COMMANDS` holds the actual command lines:

```text
mkdir -p results
set -o pipefail; python scripts/fingpt_sentiment.py 2>&1 | tee results/run.log
```

Scripts live in `scripts/` and are generic — model ids and headlines are
inputs in the script, nothing is EGX- or market-specific.

## What the pilot does

`scripts/fingpt_sentiment.py`:

1. Shallow-clones the upstream FinGPT repo inside the workflow and logs its HEAD.
2. Audits every official `FinGPT` Hugging Face checkpoint that can do
   sentiment, live against the HF API. Key fact: **the FinGPT org publishes
   LoRA adapters only — there is no standalone small FinGPT model.**
3. Attempts a real CPU `load_in_8bit=True` load of the smallest path
   (`THUDM/chatglm2-6b` + `FinGPT/fingpt-mt_chatglm2-6b_lora`) in a
   time-boxed subprocess, capturing the exact outcome.
4. Only if that load succeeds, runs sentiment generation on 3 sample
   headlines and writes `results/sentiment.json`.
5. Always writes `results/attempt.md` with the full audit and verdict.

## Checkpoints considered (pre-run audit, 2026-11-10)

| Adapter | Base model | Base gated | Base weights fp16 | ~int8 |
|---|---|---|---:|---:|
| FinGPT/fingpt-sentiment_llama2-13b_lora (dedicated sentiment) | NousResearch/Llama-2-13b-hf | mirror: no (original meta-llama: gated, manual) | ~26.0 GB | ~13 GB |
| FinGPT/fingpt-sentiment_internlm-20b_lora (dedicated sentiment) | internlm/internlm-20b | no | ~40.2 GB | ~20 GB |
| FinGPT/fingpt-mt_chatglm2-6b_lora (multi-task, incl. sentiment) | THUDM/chatglm2-6b | no | ~12.5 GB | ~6.2 GB |
| FinGPT/fingpt-mt_llama2-7b_lora | NousResearch/Llama-2-7b-hf | mirror: no | ~13.5 GB | ~6.7 GB |
| FinGPT/fingpt-mt_bloom-7b1_lora | bigscience/bloom-7b1 | no | ~14.1 GB | ~7.1 GB |
| FinGPT/fingpt-mt_falcon-7b_lora | tiiuae/falcon-7b | no | ~14.4 GB | ~7.2 GB |
| FinGPT/fingpt-mt_qwen-7b_lora | Qwen/Qwen-7B | no | ~15.4 GB | ~7.7 GB |

The LoRA adapters themselves are tiny (~12–20 MB); the base models are the
blocker. Even the smallest base is above a ~6 GB int8 CPU budget, and a
fp16/fp32 CPU load needs far more RAM/disk than a free GitHub runner offers.

## Verdict

**BLOCKED on a free CPU GitHub runner** — verified by actual dispatched runs,
not just the audit:

- Run 1: https://github.com/moos-labs/fingpt-runner/actions/runs/42823808293 — `completed / success`
- Run 2 (final, with bitsandbytes installed): https://github.com/moos-labs/fingpt-runner/actions/runs/42824221210 — `completed / success`

What actually happened in run 2 (full detail in that run's `results`
artifact, `attempt.md`):

- The upstream FinGPT repo cloned fine (HEAD `ac5c5443fbeeafad801e7a3c188e4f24cfaad4d8`).
- The live HF audit reproduced the table above exactly (smallest base:
  `THUDM/chatglm2-6b`, 12.49 GB fp16, ~6.24 GB int8).
- The ChatGLM2 tokenizer loaded OK on CPU (`ChatGLMTokenizer`).
- The base model did **not** load. With torch 2.5.1+cpu and
  bitsandbytes 0.44.1 installed, `cuda_available=False`, and
  `AutoModel.from_pretrained(..., load_in_8bit=True)` failed in
  transformers' bitsandbytes quantizer validation with exactly:

```text
RuntimeError: CUDA is required but not available for bitsandbytes.
```

- That is the only CPU-viable loading route for these bases: an unquantized
  CPU load needs the full 12.5–26 GB of weights (fp16 download, ~2× that in
  fp32 RAM) — beyond the ~6 GB int8 budget and impractical in a free
  GitHub Actions job. The FinGPT demo code itself loads with
  `device_map="cuda:0", load_in_8bit=True`, i.e. it assumes a GPU.

So: **no FinGPT sentiment inference ran, and no sentiment numbers exist.**
`results/sentiment.json` is only written if a load ever succeeds. To run
FinGPT sentiment for real, use a GPU (Colab-class is enough for the
ChatGLM2-6B / Llama-2-7B multi-task adapters in 8-bit) and load the
dedicated adapter `FinGPT/fingpt-sentiment_llama2-13b_lora` on
`NousResearch/Llama-2-13b-hf` in 8-bit, exactly as its model card shows.

## Outputs

- `results/attempt.md` — what was tried, model ids, outcome or exact blocker
- `results/run.log` — full stdout/stderr of the pilot script
- `results/load_attempt.json` — tokenizer/load outcome from the subprocess
- `results/sentiment.json` — only if inference actually ran (3 inputs + raw outputs)
