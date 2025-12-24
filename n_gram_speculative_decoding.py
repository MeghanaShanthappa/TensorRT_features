"""
Prompt Lookup Decoding (PLD) - Speculative Decoding using N-gram Matching

This script demonstrates Prompt Lookup Decoding, a speculative decoding technique
that accelerates LLM inference by using n-gram matching from the input/output
context to predict future tokens.

Key Concept:
- Instead of using a separate draft model (like traditional speculative decoding),
  PLD looks for matching n-grams in the existing context (prompt + generated text)
- When a match is found, it uses the tokens that followed that n-gram as candidates
- The main model verifies multiple candidate tokens in a single forward pass

Performance:
- Can achieve 2-4x speedup for repetitive or structured text (code, templates)
- Works best when output has patterns similar to the input
- Zero additional memory cost (no draft model needed)

Example output shows ~127 tokens/sec with PLD enabled.
"""

import copy
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation.utils import _crop_past_key_values


# =============================================================================
# Output Data Class
# =============================================================================

@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either 
            equal to `max_length` or shorter if all batches finished early due to the 
            `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*):
            Processed prediction scores of the language modeling head at each generation step.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple of attention weights for each generated token.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple of hidden states for each generated token.
    """
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# =============================================================================
# N-gram Matching for Candidate Token Prediction
# =============================================================================

@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    """
    Find candidate prediction tokens by matching n-grams in the existing context.
    
    This is the core of Prompt Lookup Decoding - we look for the last N tokens
    (n-gram) appearing earlier in the sequence, and if found, use the tokens
    that followed as our prediction candidates.
    
    Args:
        input_ids: Current token sequence [1, seq_len]
        max_ngram_size: Maximum n-gram size to match (tries largest first)
        num_pred_tokens: Number of candidate tokens to return
        
    Returns:
        Tensor of candidate tokens, or empty tensor if no match found
    """
    input_length = input_ids.size(1)

    # Validate parameters
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    # Try progressively smaller n-grams until we find a match
    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)


# =============================================================================
# Color Coding for Visualization
# =============================================================================

# ANSI color codes for visualizing speculative matches
COLORS = ["\x1b[31m", "\x1b[32m", "\x1b[34m", "\x1b[35m"]  # Red, Green, Blue, Magenta
UNDERLINE = "\x1b[4m"
RESET = "\x1b[0m"


# =============================================================================
# Prompt Lookup Decoding Implementation
# =============================================================================

@torch.no_grad()
def greedy_search_pld(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        draft_matching_window_size: int = 3,
        draft_num_candidate_tokens: int = 10,
        print_output: bool = True,
        **model_kwargs,
    ):
    """
    Greedy search with Prompt Lookup Decoding (PLD).
    
    This method extends standard greedy decoding with speculative execution:
    1. Find candidate tokens by matching n-grams in context
    2. Run forward pass on all candidates at once
    3. Accept matching tokens, reject mismatches
    4. Continue from the first mismatch point
    
    Args:
        input_ids: Input token IDs
        draft_matching_window_size: N-gram size for matching (default: 3)
        draft_num_candidate_tokens: Number of candidate tokens to predict (default: 10)
        print_output: Whether to print colored output showing matches
        **model_kwargs: Additional model arguments (attention_mask, past_key_values, etc.)
        
    Returns:
        Generated sequences (and optionally scores)
    """
    global tokenizer

    # Initialize stopping criteria and special tokens
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # Initialize scores tuple if needed
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    i = 0
    current_color_index = 0

    while True:
        i += 1
        cur_len = input_ids.shape[-1]

        # =====================================================================
        # Step 1: Find candidate tokens using n-gram matching
        # =====================================================================
        candidate_pred_tokens = find_candidate_pred_tokens(
            input_ids, 
            draft_matching_window_size, 
            draft_num_candidate_tokens
        )

        # If no match found, use a dummy token (will be rejected anyway)
        if len(candidate_pred_tokens) == 0:
            candidate_pred_tokens = torch.tensor([100], device=input_ids.device).unsqueeze(0)
        else:
            candidate_pred_tokens = candidate_pred_tokens.unsqueeze(0)
        
        # Create candidate input by appending predicted tokens
        candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

        # =====================================================================
        # Step 2: Prepare inputs and run forward pass on all candidates
        # =====================================================================
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = self._extend_attention_mask(candidate_kwargs, candidate_input_ids.shape[1])
        candidate_kwargs = self._extend_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
        
        # Single forward pass for all candidates
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # =====================================================================
        # Step 3: Verify candidates and count matches
        # =====================================================================
        new_logits = outputs.logits[:, -candidate_length - 1:]  # excludes input prompt
        selected_tokens = new_logits.argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        
        # Count how many consecutive tokens match
        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
        n_matches = min(n_matches, max_len - cur_len - 1)

        # =====================================================================
        # Step 4: Accept valid tokens and update state
        # =====================================================================
        if print_output:
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        valid_tokens = selected_tokens[:, : n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]

        # Print with color coding to show speculative matches
        if print_output:
            updated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if updated_text != current_text:
                new_text = updated_text[len(current_text):]
                if len(valid_tokens[0]) > 1:
                    # Multiple tokens accepted = speculative match (colored)
                    color = COLORS[current_color_index]
                    print(f"{color}{new_text}{RESET}", end='')
                    current_color_index = (current_color_index + 1) % len(COLORS)
                else:
                    # Single token = no speculation benefit (no color)
                    print(f"{new_text}", end='')

        # Update KV cache to match accepted tokens
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)
        model_kwargs["past_key_values"] = outputs.past_key_values

        # =====================================================================
        # Step 5: Check stopping conditions
        # =====================================================================
        if (valid_tokens == eos_token_id_tensor.item()).any():
            break
        
        if stopping_criteria(input_ids, scores):
            break

    if return_dict_in_generate:
        return GreedySearchDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
        )
    else:
        return input_ids


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """
    Demonstrate Prompt Lookup Decoding with a code modification task.
    
    This example shows PLD working well because:
    1. The output (modified code) is very similar to the input (original code)
    2. N-gram matches are likely when the model keeps most code unchanged
    3. Only small modifications are needed (adding plt.xlim)
    """
    global tokenizer
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )
    model.to(device)
    
    # Attach our PLD method to the model
    model.greedy_search_pld = greedy_search_pld.__get__(model, type(model))
    
    # Example: Code modification task (ideal for PLD - output similar to input)
    code_text = """import numpy as np
import matplotlib.pyplot as plt

# Calculate the average
average_throughput = np.mean(tokens_per_sec_arr)
print(f"Average Throughput: {average_throughput} tokens/sec")

# Plotting the histogram
plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Throughput Values')
plt.xlabel('Tokens per Second')
plt.ylabel('Frequency')
plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
plt.show()
"""
    
    question = "Can you please change x axis to start from 0"
    prompt = "[INST] Code:```python\n{code_text}``` \n\n Question: {question} \n\n Modified code:[/INST]".format(
        code_text=code_text, 
        question=question
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    # ==========================================================================
    # Generate with Prompt Lookup Decoding
    # ==========================================================================
    max_new_tokens = 500
    
    print("\n" + "="*60)
    print("Generating with Prompt Lookup Decoding (PLD)")
    print("Colored text = multiple tokens accepted via speculation")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    out = model.greedy_search_pld(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        stopping_criteria=StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)
        ]),
        draft_matching_window_size=3,
        draft_num_candidate_tokens=10,
        use_cache=True, 
        pad_token_id=0,
        eos_token_id=2,
        return_dict_in_generate=True,
        print_output=True
    )
    
    end_time = time.time()
    
    # Calculate statistics
    num_tokens_generated = len(out.sequences[0]) - len(inputs['input_ids'][0])
    total_time = end_time - start_time
    tokens_per_sec = num_tokens_generated / total_time
    
    print(f"\n\n{'='*60}")
    print(f"Performance Statistics:")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} tokens/sec")
    print(f"Total tokens generated: {num_tokens_generated}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

