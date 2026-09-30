# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Frozen from vllm-ascend 18d0537d919c303952fbe6f3442349d0de42970d,
# vllm_ascend/ops/triton/penalty.py:30 (apply_all_penalties_kernel).
# Source normalized with ast.unparse; function body and launch semantics preserved.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_seqs"])
def apply_all_penalties_kernel(logits_ptr, prompt_mask_ptr, output_mask_ptr, output_bin_counts_ptr,
                               repetition_penalties_ptr, frequency_penalties_ptr, presence_penalties_ptr, num_seqs,
                               vocab_size, stride_logits_seq, stride_logits_vocab, stride_prompt_mask_seq,
                               stride_prompt_mask_vocab, stride_output_mask_seq, stride_output_mask_vocab,
                               stride_bin_counts_seq, stride_bin_counts_vocab, BLOCK_SIZE: tl.constexpr):
    """Apply repetition, frequency, and presence penalties to logits in place."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    seqs_per_program = (num_seqs + num_programs - 1) // num_programs
    start_seq = pid * seqs_per_program
    end_seq = tl.minimum(start_seq + seqs_per_program, num_seqs)
    for seq_idx in range(start_seq, end_seq):
        repetition_penalty = tl.load(repetition_penalties_ptr + seq_idx)
        frequency_penalty = tl.load(frequency_penalties_ptr + seq_idx)
        presence_penalty = tl.load(presence_penalties_ptr + seq_idx)
        for vocab_start in range(0, vocab_size, BLOCK_SIZE):
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_SIZE)
            mask = vocab_offsets < vocab_size
            logits_offset = seq_idx * stride_logits_seq + vocab_offsets * stride_logits_vocab
            prompt_mask_offset = seq_idx * stride_prompt_mask_seq + vocab_offsets * stride_prompt_mask_vocab
            output_mask_offset = seq_idx * stride_output_mask_seq + vocab_offsets * stride_output_mask_vocab
            counts_offset = seq_idx * stride_bin_counts_seq + vocab_offsets * stride_bin_counts_vocab
            logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
            prompt_mask_val = tl.load(prompt_mask_ptr + prompt_mask_offset, mask=mask, other=False)
            output_mask_val = tl.load(output_mask_ptr + output_mask_offset, mask=mask, other=False)
            output_bin_counts = tl.load(output_bin_counts_ptr + counts_offset, mask=mask, other=0).to(tl.float32)
            need_repetition_penalty = (prompt_mask_val | output_mask_val).to(tl.int1)
            penalty_factor = tl.where(need_repetition_penalty, repetition_penalty, 1.0)
            scaling = tl.where((logits > 0.0).to(tl.int1), 1.0 / penalty_factor, penalty_factor)
            updated = logits * scaling
            updated -= frequency_penalty * output_bin_counts
            updated -= presence_penalty * output_mask_val.to(tl.float32)
            tl.store(logits_ptr + logits_offset, updated, mask=mask)
