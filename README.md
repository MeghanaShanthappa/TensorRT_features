# TensorRT_features

A collection of TensorRT-LLM and LLM inference optimization demonstrations.

## Contents

| File | Description |
|------|-------------|
| `llm_kv_cache_offloading.py` | KV cache host offloading demo (TensorRT-LLM) |
| `n_gram_speculative_decoding.py` | N-gram Speculative Decoding for faster generation (HuggingFace) |
| `faster_prefill_kv_cache_across_prompts.py` | Persistent KV cache connector for cross-instance cache sharing |

---

## 1. KV Cache Host Offloading Demo

`llm_kv_cache_offloading.py` - Demonstrates the effectiveness of KV cache host offloading in TensorRT-LLM.

### What it does

This script simulates a scenario where the GPU's KV cache is severely limited, while multiple requests with recurring prompts are processed.

### How to Run

1. **Install TensorRT-LLM:**
   ```bash
   pip install tensorrt_llm
   ```

2. **Without Offloading (baseline):**
   ```bash
   TLLM_LOG_LEVEL=DEBUG python llm_kv_cache_offloading.py 2>&1 | tee offloading_disabled.log
   ```

3. **With Offloading (optimized):**
   ```bash
   TLLM_LOG_LEVEL=DEBUG python llm_kv_cache_offloading.py --enable_offloading 2>&1 | tee offloading_enabled.log
   ```

### Expected Results

| Mode | `reused blocks` | `cache hit rate` |
|------|-----------------|------------------|
| Without Offloading | 0 | 0 |
| With Offloading | > 0 | > 0 |

### How it Works

- **Constrained GPU Cache:** Only enough for 1 request
- **Alternating Prompts:** A, B, A, B pattern forces eviction
- **With Offloading:** Evicted cache goes to host RAM instead of being discarded
- **Cache Reuse:** When prompt A returns, cache is loaded from host RAM (faster than recompute)

---

## 2. N-gram Speculative Decoding

`n_gram_speculative_decoding.py` - Demonstrates speculative decoding using n-gram matching from context.

### What it does

Prompt Lookup Decoding accelerates LLM inference by predicting future tokens based on n-gram matches in the existing context (prompt + generated text). Unlike traditional speculative decoding, it requires **no draft model**.

### Key Concept

```
Input:  "The quick brown fox jumps over the lazy dog. The quick brown..."
                                                      ^^^^^^^^^^^^^^^^
                                                      N-gram match found!
                                                      
Prediction: Use tokens that followed "The quick brown" earlier → "fox jumps over"
Verification: Run single forward pass on all candidates
Accept: Keep all matching tokens, reject from first mismatch
```

### How to Run

```bash
# Requires: transformers, torch
pip install transformers torch

# Run the demo (uses Mistral-7B)
python n_gram_speculative_decoding.py
```

### Expected Output

The script displays **colored text** showing where speculative matches succeeded:
- **Colored text** = Multiple tokens accepted via speculation (speedup!)
- **Plain text** = Single token generated (no match found)

```
Performance Statistics:
Total time: 2.37 seconds
Tokens per second: 127.55 tokens/sec
Total tokens generated: 302
```

### When PLD Works Best

| Scenario | Why it works |
|----------|--------------|
| Code modification | Output similar to input |
| Text editing | Small changes to existing text |
| Templates | Repetitive structure |
| Summarization | Quotes from original text |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_matching_window_size` | 3 | N-gram size for matching |
| `draft_num_candidate_tokens` | 10 | Number of candidates to predict |

---

## 3. Faster Prefill using KV Cache Across Prompts

`faster_prefill_kv_cache_across_prompts.py` - Demonstrates persistent KV cache sharing across LLM instances using a custom connector.

### What it does

This script implements a **KV Cache Connector** that saves computed KV cache blocks to disk and loads them back in subsequent runs, eliminating redundant computation for recurring prompts.

### Key Concept

```
Instance 1: Process prompt → Compute KV cache → Save to disk → Destroy
                                    ↓
                              [disk cache]
                                    ↓
Instance 2: Process SAME prompt → Load from disk → Skip prefill computation!
```

### What is a KV Cache Connector?

A customizable interface that allows you to:
1. **Save KV Cache:** Persist blocks to external storage (disk, database, distributed cache)
2. **Load KV Cache:** Retrieve previously computed blocks instead of recomputing
3. **Share Across Instances:** Reuse cache across different LLM instances/sessions

### How to Run

```bash
# Requires: tensorrt_llm, click
pip install tensorrt_llm click

# Run with any supported model
python faster_prefill_kv_cache_across_prompts.py meta-llama/Llama-3.1-8B-Instruct
```

### Expected Output

```
[1] Creating first LLM instance...
[2] Generating with first instance (computing KV cache)...
    KV CONNECTOR: Matched 0 blocks for request 0
    First output: <generated text>

[3] Destroying first LLM instance...
[4] Creating second LLM instance (new instance, same connector)...

[5] Generating with second instance (should load cached blocks)...
    KV CONNECTOR: Matched N blocks for request 0   ← Cache hit!
    Second output: <identical text>

✓ SUCCESS: Both outputs are identical!
```

### Key Benefits

| Benefit | Description |
|---------|-------------|
| Cross-Instance Sharing | Share computed caches across multiple LLM instances |
| Persistent Storage | Cache survives beyond the lifetime of a single instance |
| Custom Backends | Implement any storage mechanism (disk, Redis, S3, etc.) |
| Reduced Computation | Eliminate redundant prefill for repeated prompts |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache Connector                        │
├─────────────────────────────┬───────────────────────────────┤
│   Scheduler (Leader)        │   Worker                      │
├─────────────────────────────┼───────────────────────────────┤
│ • Hash token sequences      │ • Execute load/save ops       │
│ • Check disk for cache hits │ • Copy: disk ↔ GPU memory     │
│ • Schedule load operations  │ • Handle synchronization      │
│ • Schedule save operations  │                               │
└─────────────────────────────┴───────────────────────────────┘
```

