"""
KV Cache Block Priorities - Controlling Cache Retention in TensorRT-LLM

This script demonstrates how to change KV cache block priorities to control
which blocks are retained longer during cache eviction.

What is Block Priority?
    When the KV cache becomes full, TensorRT-LLM must evict blocks to make room
    for new requests. Block priority determines which blocks are evicted first:
    
    - Priority scale: 1 (lowest) to 100 (highest)
    - Default priority: 35
    - Higher priority blocks are retained longer during eviction
    
Use Cases:
    - System prompts: Set high priority (100) to keep them cached
    - Frequently reused prefixes: Higher priority for better cache hits
    - One-off queries: Lower priority, can be evicted first
    - Generated tokens: Can have different priority than prompt tokens

Key Components:
    - KvCacheRetentionConfig: Configuration object for block priorities
    - TokenRangeRetentionConfig: Set priority for specific token ranges
    - decode_retention_priority: Priority for generated (decode) tokens
    - decode_duration_ms: Optional time limit for decode token retention

Usage:
    python kv_cache_block_priorities.py
"""

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheRetentionConfig


# =============================================================================
# Example 1: Basic Generation (Default Priority)
# =============================================================================

def basic_generation():
    """
    Basic generation without custom priorities.
    All blocks are stored with default priority of 35.
    """
    print("=" * 70)
    print("Example 1: Basic Generation (Default Priority = 35)")
    print("=" * 70)
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}")
        print(f"    Generated: {generated_text!r}\n")
    
    del llm
    return outputs


# =============================================================================
# Example 2: High Priority System Prompt Tokens
# =============================================================================

def high_priority_system_prompt():
    """
    Set high priority (100) for the first 4 tokens of each prompt.
    
    This simulates a scenario where those tokens represent a system prompt
    that should be retained in cache as long as possible.
    
    Priority breakdown:
    - Tokens 0-3 (first 4): Priority 100 (highest, retained longest)
    - Remaining prompt tokens: Priority 35 (default)
    - Generated tokens: Priority 35 (default)
    """
    print("\n" + "=" * 70)
    print("Example 2: High Priority System Prompt (First 4 Tokens = 100)")
    print("=" * 70)
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Configure priority for first 4 tokens
    # TokenRangeRetentionConfig(start_token, end_token, priority, duration_ms)
    # - start_token: 0 (first token)
    # - end_token: 4 (exclusive, so tokens 0, 1, 2, 3)
    # - priority: 100 (highest)
    # - duration_ms: None (never expires)
    token_range_config = KvCacheRetentionConfig.TokenRangeRetentionConfig(
        0,      # start_token
        4,      # end_token (exclusive)
        100,    # priority (highest)
        None    # duration_ms (never expires)
    )
    
    kv_cache_retention_config = KvCacheRetentionConfig(
        token_range_retention_configs=[token_range_config],
        decode_retention_priority=35,  # Generated tokens get default priority
        decode_duration_ms=None        # No time limit
    )
    
    print("\nRetention Config:")
    print(f"  - Tokens 0-3: Priority 100 (system prompt)")
    print(f"  - Other tokens: Priority 35 (default)")
    print(f"  - Generated tokens: Priority 35")
    print()
    
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        kv_cache_retention_config=kv_cache_retention_config
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}")
        print(f"    Generated: {generated_text!r}\n")
    
    del llm
    return outputs


# =============================================================================
# Example 3: Per-Prompt Priority Configuration
# =============================================================================

def per_prompt_priorities():
    """
    Different priority configurations for different prompts.
    
    This demonstrates how to provide a list of retention configs,
    one for each prompt, allowing fine-grained control.
    """
    print("\n" + "=" * 70)
    print("Example 3: Per-Prompt Priority Configuration")
    print("=" * 70)
    
    prompts = [
        "Hello, my name is",           # High priority (important user)
        "The president of the US is",   # Medium priority
        "The capital of France is",     # Low priority (one-off query)
        "The future of AI is",          # High priority (frequently asked)
    ]
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Create different configs for each prompt
    configs = []
    priorities = [100, 50, 20, 100]  # Different priority for each prompt
    
    for i, priority in enumerate(priorities):
        # Set all tokens in the prompt to the specified priority
        token_range_config = KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0,          # start_token
            1000,       # end_token (large number to cover all tokens)
            priority,   # priority
            None        # duration_ms
        )
        
        config = KvCacheRetentionConfig(
            token_range_retention_configs=[token_range_config],
            decode_retention_priority=priority,  # Match decode to prompt priority
            decode_duration_ms=None
        )
        configs.append(config)
    
    print("\nPer-Prompt Retention Configs:")
    for i, (prompt, priority) in enumerate(zip(prompts, priorities)):
        print(f"  [{i}] Priority {priority:3d}: {prompt!r}")
    print()
    
    # Pass list of configs (must match length of prompts)
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        kv_cache_retention_config=configs  # List of configs
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Priority {priorities[i]:3d} | {prompt!r}")
        print(f"    Generated: {generated_text!r}\n")
    
    del llm
    return outputs


# =============================================================================
# Example 4: Time-Limited Cache Retention
# =============================================================================

def time_limited_retention():
    """
    Demonstrate time-limited cache retention for decode tokens.
    
    This is useful when you want generated tokens to be available for
    reuse for a limited time, then automatically deprioritized.
    """
    print("\n" + "=" * 70)
    print("Example 4: Time-Limited Cache Retention")
    print("=" * 70)
    
    prompts = [
        "Explain machine learning in simple terms:",
    ]
    sampling_params = SamplingParams(max_tokens=64)

    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # High priority prompt, time-limited decode tokens
    prompt_config = KvCacheRetentionConfig.TokenRangeRetentionConfig(
        0,      # start_token
        100,    # end_token
        100,    # priority (highest for prompt)
        None    # no time limit for prompt
    )
    
    kv_cache_retention_config = KvCacheRetentionConfig(
        token_range_retention_configs=[prompt_config],
        decode_retention_priority=50,     # Medium priority for decode
        decode_duration_ms=5000           # 5 seconds before priority drops
    )
    
    print("\nRetention Config:")
    print(f"  - Prompt tokens: Priority 100 (permanent)")
    print(f"  - Generated tokens: Priority 50 for 5 seconds, then deprioritized")
    print()
    
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        kv_cache_retention_config=kv_cache_retention_config
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}\n")
    
    del llm
    return outputs


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Run all examples demonstrating KV cache block priority control.
    """
    print("\n" + "=" * 70)
    print("KV Cache Block Priorities Demo")
    print("=" * 70)
    print("""
    Block priority determines which KV cache blocks are evicted first
    when the cache is full:
    
    Priority Scale: 1 (lowest) ───────────────────── 100 (highest)
                    ↓                                      ↓
                Evicted first                    Retained longest
    
    Default Priority: 35
    """)
    
    # Run examples
    basic_generation()
    high_priority_system_prompt()
    per_prompt_priorities()
    time_limited_retention()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()



