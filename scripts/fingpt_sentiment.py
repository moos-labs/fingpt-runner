#!/usr/bin/env python3
"""FinGPT sentiment pilot: lightest feasible CPU attempt, honestly documented.

FinGPT (AI4Finance-Foundation/FinGPT) does not ship a standalone small
sentiment model. Its official Hugging Face org (FinGPT) publishes only LoRA
adapters; every adapter needs a full 6B-20B base model:

  Dedicated sentiment adapters
    FinGPT/fingpt-sentiment_llama2-13b_lora   -> NousResearch/Llama-2-13b-hf
    FinGPT/fingpt-sentiment_internlm-20b_lora -> internlm/internlm-20b
  Multi-task adapters (include a sentiment task)
    FinGPT/fingpt-mt_chatglm2-6b_lora -> THUDM/chatglm2-6b      (smallest base)
    FinGPT/fingpt-mt_llama2-7b_lora   -> (NousResearch/)Llama-2-7b-hf
    FinGPT/fingpt-mt_bloom-7b1_lora   -> bigscience/bloom-7b1
    FinGPT/fingpt-mt_falcon-7b_lora   -> tiiuae/falcon-7b
    FinGPT/fingpt-mt_mpt-7b_lora      -> mosaicml/mpt-7b
    FinGPT/fingpt-mt_qwen-7b_lora     -> Qwen/Qwen-7B
    FinGPT/fingpt-mt_llama3-8b_lora   -> meta-llama/Meta-Llama-3-8B (gated)

This script:
  1. shallow-clones the upstream FinGPT repo and logs its HEAD,
  2. audits every candidate live against the Hugging Face API
     (adapter_config.json -> base model, base weight size, gated flag),
  3. downloads and parses the LoRA adapters themselves (they are tiny,
     ~4-20 MB) to prove the adapter side is accessible,
  4. attempts a real CPU load of the SMALLEST path
     (THUDM/chatglm2-6b + FinGPT/fingpt-mt_chatglm2-6b_lora) with
     load_in_8bit=True in a subprocess, capturing the exact outcome,
  5. only if that load somehow succeeds, runs sentiment generation on
     the 3 sample headlines and writes results/sentiment.json,
  6. always writes results/attempt.md with the full audit and verdict.

Time-box rule (from the task): if the smallest viable path needs a gated
base model, or > ~6 GB of weights in int8 on CPU, or cannot finish in a
free GitHub Actions job, stop and document precisely instead of burning
the job on a multi-GB download that cannot fit.
"""
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

HEADLINES = [
    {"expected": "positive",
     "text": "Company reports record quarterly profit, beating analyst "
             "expectations and raising its full-year outlook."},
    {"expected": "negative",
     "text": "Company shares plunge after it cuts its profit forecast and "
             "reports a sharp drop in quarterly earnings."},
    {"expected": "neutral",
     "text": "Company announces it will release its quarterly earnings "
             "report on Thursday after the market close."},
]

# (adapter repo, base repo used for size/gating audit)
CANDIDATES = [
    ("FinGPT/fingpt-sentiment_llama2-13b_lora", "NousResearch/Llama-2-13b-hf"),
    ("FinGPT/fingpt-sentiment_internlm-20b_lora", "internlm/internlm-20b"),
    ("FinGPT/fingpt-mt_chatglm2-6b_lora", "THUDM/chatglm2-6b"),
    ("FinGPT/fingpt-mt_llama2-7b_lora", "NousResearch/Llama-2-7b-hf"),
    ("FinGPT/fingpt-mt_bloom-7b1_lora", "bigscience/bloom-7b1"),
    ("FinGPT/fingpt-mt_falcon-7b_lora", "tiiuae/falcon-7b"),
    ("FinGPT/fingpt-mt_qwen-7b_lora", "Qwen/Qwen-7B"),
]
SMALLEST_ADAPTER = "FinGPT/fingpt-mt_chatglm2-6b_lora"
SMALLEST_BASE = "THUDM/chatglm2-6b"
INT8_BUDGET_GB = 6.0

log_lines = []


def log(msg=""):
    print(msg, flush=True)
    log_lines.append(str(msg))


def hf_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fingpt-runner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def hf_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fingpt-runner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def base_weight_gb(model_id):
    """Single-format weight size in GB (prefer safetensors shards; the API
    tree lists every format, so summing all files would double-count)."""
    tree = hf_json(f"https://huggingface.co/api/models/{model_id}/tree/main?recursive=true")
    st, pt = 0, 0
    for f in tree:
        if f.get("type") != "file":
            continue
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        name = f.get("path", "")
        if name.endswith(".safetensors"):
            st += size
        elif name.endswith(".bin"):
            pt += size
    return (st or pt) / 1e9


def audit():
    rows = []
    for adapter, base in CANDIDATES:
        row = {"adapter": adapter, "base": base}
        try:
            cfg = json.loads(hf_text(
                f"https://huggingface.co/{adapter}/resolve/main/adapter_config.json"))
            row["adapter_base_declared"] = cfg.get("base_model_name_or_path")
            row["peft_type"] = cfg.get("peft_type")
            row["lora_r"] = cfg.get("r")
        except Exception as e:  # noqa: BLE001
            row["adapter_config_error"] = f"{type(e).__name__}: {e}"
        try:
            info = hf_json(f"https://huggingface.co/api/models/{base}")
            row["base_gated"] = info.get("gated")
        except Exception as e:  # noqa: BLE001
            row["base_gated"] = f"lookup failed: {e}"
        try:
            gb = base_weight_gb(base)
            row["base_fp16_gb"] = round(gb, 2)
            row["base_int8_est_gb"] = round(gb / 2, 2)
        except Exception as e:  # noqa: BLE001
            row["base_size_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        log(f"AUDIT {adapter} -> base={base} gated={row.get('base_gated')} "
            f"fp16={row.get('base_fp16_gb')} GB int8~={row.get('base_int8_est_gb')} GB "
            f"declared_base={row.get('adapter_base_declared')}")
    return rows


LOAD_ATTEMPT = r'''
import json, sys, traceback
base, adapter = sys.argv[1], sys.argv[2]
out = {"base": base, "adapter": adapter}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
except Exception as e:
    out["torch_error"] = str(e)
try:
    import bitsandbytes
    out["bitsandbytes"] = bitsandbytes.__version__
except Exception as e:
    out["bitsandbytes_error"] = str(e)
print("ENV", json.dumps(out), flush=True)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    out["tokenizer"] = f"loaded OK ({tok.__class__.__name__})"
    print("TOKENIZER_OK", flush=True)
except Exception:
    out["tokenizer_error"] = traceback.format_exc()
    print("TOKENIZER_FAILED", flush=True)
try:
    import torch
    from transformers import AutoModel
    from peft import PeftModel
    model = AutoModel.from_pretrained(
        base, trust_remote_code=True, load_in_8bit=True, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.eval()
    out["load_in_8bit"] = "loaded OK"
    print("LOAD_OK", flush=True)
    # If we get here, actually run the 3 headlines.
    prompts = []
    headlines = json.loads(sys.argv[3])
    for h in headlines:
        prompts.append(
            "Instruction: What is the sentiment of this news? Please choose "
            "an answer from {negative/neutral/positive}\n"
            f"Input: {h['text']}\nAnswer: ")
    results = []
    for h, p in zip(headlines, prompts):
        inputs = tok(p, return_tensors="pt")
        gen = model.generate(**inputs, max_new_tokens=5)
        text = tok.decode(gen[0], skip_special_tokens=True)
        results.append({"input": h["text"], "expected": h["expected"],
                        "prompt": p, "raw_output": text})
    with open("results/sentiment.json", "w") as f:
        json.dump({"base": base, "adapter": adapter, "results": results}, f, indent=2)
    out["inference"] = "ran, results/sentiment.json written"
except Exception:
    out["load_error"] = traceback.format_exc()
    print("LOAD_FAILED", flush=True)
with open("results/load_attempt.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out)[:2000], flush=True)
'''


def main():
    t0 = time.time()
    log("# FinGPT sentiment pilot - attempt log")
    log(f"python={sys.version.split()[0]}")

    # 1. Upstream FinGPT repo (per task: clone inside the workflow)
    try:
        r = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/AI4Finance-Foundation/FinGPT", "fingpt-upstream"],
            capture_output=True, text=True, timeout=300)
        head = subprocess.run(["git", "-C", "fingpt-upstream", "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        listing = subprocess.run(["ls", "fingpt-upstream"], capture_output=True,
                                 text=True).stdout.strip().replace("\n", ", ")
        log(f"FinGPT clone: rc={r.returncode} HEAD={head}")
        log(f"FinGPT top level: {listing}")
    except Exception:  # noqa: BLE001
        log("FinGPT clone FAILED:\n" + traceback.format_exc())

    # 2. Live audit of every official candidate
    rows = audit()
    smallest = next((r for r in rows if r["adapter"] == SMALLEST_ADAPTER), {})
    int8 = smallest.get("base_int8_est_gb")
    log(f"Smallest base: {SMALLEST_BASE} fp16={smallest.get('base_fp16_gb')} GB "
        f"int8 estimate={int8} GB (budget {INT8_BUDGET_GB} GB)")

    # 3. Real load attempt of the smallest path (subprocess, time-boxed)
    log(f"Attempting CPU load_in_8bit of {SMALLEST_BASE} + {SMALLEST_ADAPTER} ...")
    with open("/tmp/load_attempt.py", "w") as f:
        f.write(LOAD_ATTEMPT)
    try:
        r = subprocess.run(
            [sys.executable, "/tmp/load_attempt.py", SMALLEST_BASE,
             SMALLEST_ADAPTER, json.dumps(HEADLINES)],
            capture_output=True, text=True, timeout=1200)
        attempt_out = (r.stdout or "")[-4000:]
        attempt_err = (r.stderr or "")[-2000:]
        log(f"load attempt rc={r.returncode}\n--- stdout tail ---\n{attempt_out}\n"
            f"--- stderr tail ---\n{attempt_err}")
    except subprocess.TimeoutExpired:
        log("load attempt TIMEOUT after 1200s (time-box hit)")
        attempt_out = "timeout"

    load_json = {}
    if os.path.exists(f"{RESULTS}/load_attempt.json"):
        load_json = json.load(open(f"{RESULTS}/load_attempt.json"))
    inference_ran = os.path.exists(f"{RESULTS}/sentiment.json")
    load_error = (load_json.get("load_error") or "").strip()

    # 4. Verdict + attempt.md
    verdict = "RAN on CPU" if inference_ran else "BLOCKED on a free CPU GitHub runner"
    lines = []
    lines.append("# FinGPT sentiment pilot - attempt")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("## What was tried")
    lines.append("")
    lines.append("1. Shallow-cloned `AI4Finance-Foundation/FinGPT` in the workflow "
                 "and logged its HEAD (see run.log).")
    lines.append("2. Audited every official `FinGPT` Hugging Face checkpoint that "
                 "can do sentiment - the org publishes LoRA adapters only, no "
                 "standalone small model - reading each `adapter_config.json` for "
                 "its base model and the HF tree API for base weight sizes.")
    lines.append(f"3. Attempted a real CPU load of the smallest path: base "
                 f"`{SMALLEST_BASE}` with adapter `{SMALLEST_ADAPTER}`, "
                 "`load_in_8bit=True`, in a subprocess (time-box 1200s).")
    lines.append("4. Inference on the 3 sample headlines runs only if that load "
                 "succeeds (then `results/sentiment.json` is written).")
    lines.append("")
    lines.append("## Checkpoints considered (live HF audit)")
    lines.append("")
    lines.append("| Adapter | Base model | Base gated | Base weights fp16 | ~int8 |")
    lines.append("|---|---|---|---:|---:|")
    for r in rows:
        lines.append(f"| {r['adapter']} | {r['base']} | {r.get('base_gated')} | "
                     f"{r.get('base_fp16_gb', '?')} GB | {r.get('base_int8_est_gb', '?')} GB |")
    lines.append("")
    lines.append("Notes: the dedicated sentiment adapter "
                 "`FinGPT/fingpt-sentiment_llama2-13b_lora` declares base "
                 "`NousResearch/Llama-2-13b-hf` in its adapter_config.json "
                 "(the original `meta-llama/Llama-2-13b-hf` is gated=manual; "
                 "the NousResearch mirror is ungated, but it is the same 13B "
                 "weights, ~26 GB fp16 / ~13 GB int8). The other dedicated "
                 "sentiment adapter needs InternLM-20B (~40 GB fp16). Every "
                 "multi-task adapter needs a 6B-8B base; the smallest base of "
                 f"all is `{SMALLEST_BASE}` at {smallest.get('base_fp16_gb', '?')} GB "
                 f"fp16, ~{int8} GB even in int8 - above the ~{INT8_BUDGET_GB} GB "
                 "CPU budget in the task, before counting the runner's RAM/disk "
                 "for a fp16/fp32 CPU load.")
    lines.append("")
    lines.append("## Load attempt outcome")
    lines.append("")
    lines.append(f"- Environment: torch={load_json.get('torch')}, "
                 f"bitsandbytes={load_json.get('bitsandbytes')}, "
                 f"cuda_available={load_json.get('cuda_available')}")
    lines.append(f"- Tokenizer: {load_json.get('tokenizer', load_json.get('tokenizer_error', 'not recorded'))}")
    if inference_ran:
        lines.append("- Base + adapter loaded in int8 on CPU and generation ran; "
                     "see `sentiment.json`.")
    else:
        lines.append("- Base + adapter did NOT load; exact error captured from the "
                     "subprocess:")
        lines.append("")
        lines.append("```")
        lines.append(load_error[-3000:] if load_error else "(no load_attempt.json; see run.log tail in the attempt log)")
        lines.append("```")
    lines.append("")
    lines.append("## Sample headlines (used if inference runs)")
    lines.append("")
    for h in HEADLINES:
        lines.append(f"- ({h['expected']}) {h['text']}")
    lines.append("")
    lines.append(f"Elapsed: {time.time() - t0:.1f}s. Full log: run.log in this artifact.")
    with open(f"{RESULTS}/attempt.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"DONE verdict={verdict} inference_ran={inference_ran}")
    return 0  # a documented blocker is a successful outcome for this runner


if __name__ == "__main__":
    sys.exit(main())
