# TensorRT_features

A collection of TensorRT-LLM feature demonstrations and examples.

## KV Cache Host Offloading Demo

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

