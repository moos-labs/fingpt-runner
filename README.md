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

_Pending the first dispatched run — this section is updated from the actual
run artifact (`results/attempt.md`), not from the pre-run audit alone._

## Outputs

- `results/attempt.md` — what was tried, model ids, outcome or exact blocker
- `results/run.log` — full stdout/stderr of the pilot script
- `results/load_attempt.json` — tokenizer/load outcome from the subprocess
- `results/sentiment.json` — only if inference actually ran (3 inputs + raw outputs)
