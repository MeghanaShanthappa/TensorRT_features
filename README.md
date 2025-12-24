# TensorRT_features

A collection of TensorRT-LLM and LLM inference optimization demonstrations.

## Contents

| File | Description |
|------|-------------|
| `llm_kv_cache_offloading.py` | KV cache host offloading demo (TensorRT-LLM) |
| `n_gram_speculative_decoding.py` | N-gram Speculative Decoding for faster generation (HuggingFace) |

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

