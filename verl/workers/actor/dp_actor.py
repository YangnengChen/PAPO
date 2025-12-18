# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig
from torch.distributions import Categorical


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    # def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    #     """
    #     Returns:
    #         log_probs: # (bs, response_len)
    #     """
    #     input_ids = micro_batch["input_ids"]
    #     batch_size, seqlen = input_ids.shape
    #     attention_mask = micro_batch["attention_mask"]
    #     position_ids = micro_batch["position_ids"]
    #     responses = micro_batch["responses"]
    #     response_length = responses.size(-1)
    #     if position_ids.dim() == 3:  # qwen2vl mrope
    #         position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

    #     multi_modal_inputs = defaultdict(list)
    #     if "multi_modal_inputs" in micro_batch:
    #         for input_dict in micro_batch["multi_modal_inputs"]:
    #             for key, value in input_dict.items():
    #                 multi_modal_inputs[key].append(value)

    #         for key, value in multi_modal_inputs.items():
    #             if len(value) != 0:
    #                 multi_modal_inputs[key] = torch.cat(value, dim=0)
    #             else:
    #                 multi_modal_inputs[key] = None

    #     if self.config.padding_free:
    #         input_ids_rmpad, indices, *_ = unpad_input(
    #             input_ids.unsqueeze(-1), attention_mask
    #         )  # input_ids_rmpad (total_nnz, ...)
    #         input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    #         # unpad the position_ids to align the rotary
    #         if position_ids.dim() == 3:
    #             position_ids_rmpad = (
    #                 index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
    #                 .transpose(0, 1)
    #                 .unsqueeze(1)
    #             )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
    #         else:
    #             position_ids_rmpad = index_first_axis(
    #                 rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    #             ).transpose(0, 1)

    #         # for compute the log_prob
    #         input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

    #         # pad and slice the inputs if sp > 1
    #         if self.config.ulysses_size > 1:
    #             input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
    #                 input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
    #             )
    #             input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
    #                 input_ids_rmpad_rolled, None, self.config.ulysses_size
    #             )

    #         input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

    #         # only pass input_ids and position_ids to enable flash_attn_varlen
    #         output = self.actor_module(
    #             input_ids=input_ids_rmpad,
    #             attention_mask=None,
    #             position_ids=position_ids_rmpad,
    #             **multi_modal_inputs,
    #             use_cache=False,
    #         )  # prevent model thinks we are generating
    #         logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    #         logits_rmpad.div_(temperature)
    #         # ((total_nnz / sp) + pad)
    #         log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

    #         # gather log_prob if sp > 1
    #         if self.config.ulysses_size > 1:
    #             # gather and unpad for the ulysses sp
    #             log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

    #         # pad back to (bsz, seqlen)
    #         full_log_probs = pad_input(
    #             hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    #         )
    #         log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
    #     else:
    #         output = self.actor_module(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             **multi_modal_inputs,
    #             use_cache=False,
    #         )
    #         logits: torch.Tensor = output.logits
    #         logits.div_(temperature)
    #         logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
    #         log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

    #     return log_probs
    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            # if getattr(self.config, 'use_vppo_on_entropy', False):
            #     dist = Categorical(logits=logits_rmpad)
            #     entropy = dist.entropy()

            #     if self.config.ulysses_size > 1:
            #         entropy = gather_outputs_and_unpad(entropy, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            #     full_entropy = pad_input(
            #         hidden_states=entropy.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            #     )
            #     entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
            # else:
            #     entropy = torch.zeros_like(log_probs)
            
            dist = Categorical(logits=logits_rmpad)
            entropy = dist.entropy()

            if self.config.ulysses_size > 1:
                entropy = gather_outputs_and_unpad(entropy, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_entropy = pad_input(
                hidden_states=entropy.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

            # if getattr(self.config, 'use_vppo_on_entropy', False):
            #     dist = Categorical(logits=logits)
            #     entropy = dist.entropy()
            # else:
            #     entropy = torch.zeros_like(log_probs)
            dist = Categorical(logits=logits)
            entropy = dist.entropy()
            

        # Use vppo based on entropy: Return a dictionary
        return {"log_probs": log_probs, "entropy": entropy}

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)['log_probs']
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # for contrastive kl
        if "aug_log_probs" in data.batch.keys() and self.config.use_kl_prcp:
            select_keys.append("aug_log_probs")
            non_tensor_select_keys.append("kl_prcp_weighting")
            non_tensor_select_keys.append("kl_prcp_coef")
        
        if self.config.use_sft_loss:
            non_tensor_select_keys.append("correctness_mult_mask")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    # for kl prcp
                    kl_prcp_weighting = micro_batch.non_tensor_batch.pop("kl_prcp_weighting", None)
                    kl_prcp_coef = micro_batch.non_tensor_batch.pop("kl_prcp_coef", None)

                    # for sft loss
                    correctness_mult_mask = micro_batch.non_tensor_batch.pop("correctness_mult_mask", None)

                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    output = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs = output['log_probs']
                    entropy = output['entropy']
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss
                    # entropy_loss = average_loss(entropy, response_mask, mode=self.config.loss_avg_mode)
                    
                    loss_token_mask = None # Default to None

                    if self.config.use_vppo_on_entropy:
                        # Use vppo based on entropy at the response level.
                        # For each response, select the top p% of tokens with the highest entropy.
                        top_p = self.config.top_p_entropy_tokens
                        
                        # Calculate the number of tokens to keep for each response in the batch.
                        # 'k' will be a tensor of shape (mini_batch_size,).
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()

                        # To ensure padded tokens are not selected, we mask their entropy to a very low value.
                        masked_entropy = entropy.clone()
                        masked_entropy[~response_mask.bool()] = -float('inf')

                        # Sort entropies in descending order to find the top-k values and their original indices.
                        sorted_entropy_vals, sorted_indices = torch.sort(masked_entropy, dim=1, descending=True)
                        
                        # Create a tensor representing the rank of each token within its response.
                        range_tensor = torch.arange(entropy.size(1), device=entropy.device).expand_as(entropy)

                        # Create a mask to identify the top k ranked tokens for each response.
                        # k is unsqueezed to (mini_batch_size, 1) to enable broadcasting.
                        rank_mask = range_tensor < k.unsqueeze(1)

                        # Scatter the rank_mask back to the original token order to create the final mask.
                        top_p_mask = torch.zeros_like(entropy, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        
                        loss_token_mask = top_p_mask.to(entropy.dtype)
                        
                        # The subsequent logging code expects a single `threshold` value.
                        # We will calculate the threshold for each response (the k-th largest entropy)
                        # and then average them for logging purposes.
                        
                        # To avoid index errors for responses where k=0, clamp k at 1 for indexing.
                        k_safe_for_indexing = k.clone().clamp(min=1)
                        threshold_indices = (k_safe_for_indexing - 1).unsqueeze(1)
                        
                        # Gather the threshold value for each response from the sorted entropy values.
                        threshold_per_response = torch.gather(sorted_entropy_vals, 1, threshold_indices.long()).squeeze(1)
                        
                        # For responses where k was originally 0, the threshold is not meaningful, so we ignore them.
                        valid_thresholds = threshold_per_response[k > 0]
                        if valid_thresholds.numel() > 0:
                            threshold = valid_thresholds.mean()
                        else:
                            # Fallback if no tokens are selected in any response.
                            threshold = torch.tensor(0.0, device=entropy.device)

                        # Add logging.
                        with torch.no_grad():
                            num_total_valid_tokens = response_mask.sum()
                            num_selected_tokens = top_p_mask.sum()
                            
                            if num_total_valid_tokens > 0:
                                # Log the actual fraction of tokens used.
                                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                                metrics["actor/entropy_token_fraction"].append(actual_token_fraction)
                                
                                # Log the entropy threshold to observe its changes during training.
                                metrics["actor/entropy_threshold"].append(threshold.item())

                                # Log mean entropy for selected tokens.
                                selected_entropies = torch.masked_select(entropy, top_p_mask.bool())
                                if selected_entropies.numel() > 0:
                                    metrics["actor/entropy_mean_selected"].append(selected_entropies.mean().item())
                                    
                                # Log mean entropy for rejected tokens for comparison.
                                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                                rejected_entropies = torch.masked_select(entropy, rejected_mask)
                                if rejected_entropies.numel() > 0:
                                    metrics["actor/entropy_mean_rejected"].append(rejected_entropies.mean().item())

                    if self.config.use_vppo_on_perception:
                        # Use vppo based on perception: Calculate the top p tokens based on perception.
                        top_p = self.config.top_p_perception_tokens
                        
                        aug_log_probs = model_inputs["aug_log_probs"]
                        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=10.0)

                        low_var_kl_for_sort = low_var_kl.clone()
                        invalid_mask = ~response_mask.bool()
                        low_var_kl_for_sort[invalid_mask] = -torch.inf

                        # Calculate the number of tokens to keep for each response.
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()

                        # Sort the perception differences in descending order to get values and original indices.
                        sorted_vals, sorted_indices = torch.sort(low_var_kl_for_sort, dim=1, descending=True)
                        
                        # Create a rank mask to identify the top k positions in each response.
                        range_tensor = torch.arange(low_var_kl_for_sort.size(1), device=low_var_kl_for_sort.device).expand_as(low_var_kl_for_sort)
                        rank_mask = range_tensor < k.unsqueeze(1)

                        # Use scatter to map the rank mask back to the original token order, creating the final perception mask.
                        top_p_mask = torch.zeros_like(low_var_kl_for_sort, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        
                        top_p_mask = (top_p_mask.bool() & response_mask.bool()).to(log_probs.dtype)

                        if loss_token_mask is not None:
                            loss_token_mask = (loss_token_mask.bool() | top_p_mask.bool()).to(log_probs.dtype)
                        else:
                            loss_token_mask = top_p_mask

                        # Calculate an average threshold for logging purposes.
                        # This is the mean of the k-th largest perception difference for each response.
                        k_safe_for_indexing = k.clone().clamp(min=1)
                        threshold_indices = (k_safe_for_indexing - 1).unsqueeze(1)
                        
                        threshold_per_response = torch.gather(sorted_vals, 1, threshold_indices.long()).squeeze(1)
                        
                        valid_thresholds = threshold_per_response[k > 0]
                        if valid_thresholds.numel() > 0:
                            threshold = valid_thresholds.mean()
                        else:
                            threshold = torch.tensor(0.0, device=low_var_kl_for_sort.device)

                        # Add logging.
                        with torch.no_grad():
                            num_total_valid_tokens = response_mask.sum()
                            num_selected_tokens = top_p_mask.sum()
                            
                            if num_total_valid_tokens > 0:
                                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                                metrics["actor/perception_token_fraction"].append(actual_token_fraction)
                                metrics["actor/low_var_kl_threshold"].append(threshold.item())

                                selected_low_var_kl = torch.masked_select(low_var_kl, top_p_mask.bool())
                                if selected_low_var_kl.numel() > 0:
                                    metrics["actor/low_var_kl_mean_selected"].append(selected_low_var_kl.mean().item())

                                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                                rejected_low_var_kl = torch.masked_select(low_var_kl, rejected_mask)
                                if rejected_low_var_kl.numel() > 0:
                                    metrics["actor/low_var_kl_mean_rejected"].append(rejected_low_var_kl.mean().item())

                    if self.config.use_vppo_on_entropy and self.config.use_vppo_on_perception:
                        # Add combined logging.
                        metrics["actor/combined_token_fraction"].append(
                            (loss_token_mask.sum() / response_mask.sum()).item()
                        )
                        metrics["actor/combined_entropy_mean_selected"].append(
                            torch.masked_select(entropy, loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_entropy_mean_rejected"].append(
                            torch.masked_select(entropy, response_mask.bool() & ~loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_perception_mean_selected"].append(
                            torch.masked_select(low_var_kl, loss_token_mask.bool()).mean().item()
                        )
                        metrics["actor/combined_perception_mean_rejected"].append(
                            torch.masked_select(low_var_kl, response_mask.bool() & ~loss_token_mask.bool()).mean().item()
                        )
                    
                    
                    if self.config.use_entopy_advantage_shaping:
                        
                        advantages += advantages * torch.min(self.config.entropy_alpha * entropy.detach(), advantages.abs() / self.config.entropy_kappa)
                        
                        
                    elif self.config.use_VD_advantage_shaping:
                        aug_log_probs = model_inputs["aug_log_probs"]
                        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=5.0)
                        advantages += advantages * torch.min(self.config.kl_alpha * low_var_kl.detach(), advantages.abs() / self.config.kl_kappa)
                    elif self.config.use_combined_advantage_shaping:
    
                        # -----------------------------------------------------------
                        # 1. 准备 Visual Dependency (KL) 指标
                        # -----------------------------------------------------------
                        aug_log_probs = model_inputs["aug_log_probs"]
                        # 计算 log 概率差值并截断，防止数值溢出
                        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                        # 使用 exp(x) - x - 1 近似计算低方差 KL 散度
                        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                        # 截断 KL 值，防止单个样本影响过大
                        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=5.0)

                        # -----------------------------------------------------------
                        # 2. 计算加权得分 (Weighted Score)
                        # -----------------------------------------------------------
                        # 注意：这里我们使用 detach()，因为我们只用这些指标来调整 Advantage，
                        # 而不需要通过 Advantage 的梯度反向传播来直接优化 Entropy 或 KL。
                        
                        # 权重1: 熵 (代表探索欲望)
                        entropy_term = self.config.entropy_alpha * entropy.detach()
                        
                        # 权重2: Visual Dependency / KL (代表与增强图像的一致性/鲁棒性)
                        vd_term = self.config.kl_alpha * low_var_kl.detach()
                        
                        # 加权求和得到总得分
                        combined_score = entropy_term + vd_term

                        # -----------------------------------------------------------
                        # 3. 统一进行 Advantage Shaping
                        # -----------------------------------------------------------
                        # 这里需要一个统一的 kappa 参数。
                        # 建议在 config 中添加 self.config.shaping_kappa，
                        # 如果没有，暂时可以用 self.config.entropy_kappa 代替。
                        shaping_kappa = getattr(self.config, 'shaping_kappa', self.config.entropy_kappa)

                        # 计算修正项：
                        # 逻辑是：Advantage 增强幅度不应超过 combined_score，
                        # 同时受到 abs(advantages) / kappa 的相对约束
                        shaping_factor = torch.min(
                            combined_score, 
                            advantages.abs() / shaping_kappa
                        )

                        # 应用修正
                        advantages += advantages * shaping_factor 

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                        loss_token_mask=loss_token_mask,
                        entropy=entropy,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    discount_ratio = 1.0 # for entropy losses; maybe updated by annealing kl_prcp settings
                    
                    # for kl_prcp
                    if "aug_log_probs" in model_inputs:
                        aug_log_probs = model_inputs["aug_log_probs"]
                        aug_entropy_loss = -VF.masked_mean(aug_log_probs, response_mask)  # estimator of entropy loss
                        
                        # compute kl_prcp
                        aug_kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=aug_log_probs,
                            kl_penalty=self.config.kl_prcp_penalty,
                        )
                        
                        # turn contastive_kl_weighting to tensor
                        kl_prcp_weighting = torch.tensor(kl_prcp_weighting, device=aug_kld.device, dtype=aug_kld.dtype)
                        kl_prcp_weighting = kl_prcp_weighting.unsqueeze(1) # (bsz, 1)
                        aug_kld = aug_kld * kl_prcp_weighting

                        # if add additional token-level masking
                        if self.config.use_kl_prcp_token_level_mask:
                            top_p = self.config.kl_prcp_token_level_mask_top_p
                            assert top_p > 0.0 and top_p < 1.0, "top_p must be in (0, 1) for token-level masking"
                            # --- Only keep *valid* tokens when computing the per-sample threshold -------------
                            valid_mask = response_mask.bool()                     # (bsz, resp_len)
                            masked_kld = aug_kld.masked_fill(~valid_mask, float("nan"))
                            thresh = torch.nanquantile(masked_kld, 1.0 - top_p, dim=1, keepdim=True)
                            token_mask = aug_kld >= thresh          # (bsz, resp_len)  bool
                            # Apply the mask (cast to same dtype)
                            aug_kld = aug_kld * token_mask.to(aug_kld.dtype)

                        # if add kl_prcp clipping
                        if self.config.use_kl_prcp_clipping:
                            aug_kld = torch.clamp(aug_kld, min=0.0, max=self.config.kl_prcp_clipping)

                        # kl_prcp loss
                        kl_prcp_loss = average_loss(aug_kld, response_mask, mode=self.config.loss_avg_mode)

                        if kl_prcp_coef is None:
                            kl_prcp_coef = self.config.kl_prcp_coef
                        else:
                            kl_prcp_coef = kl_prcp_coef[0] # an array of identical values, take the first one
                        
                        # if annealing; applying the same discount to aug_entropy_loss
                        if kl_prcp_coef != self.config.kl_prcp_coef:
                            discount_ratio = kl_prcp_coef / self.config.kl_prcp_coef

                        pg_loss = pg_loss - kl_prcp_loss * kl_prcp_coef

                        # if adding masked entropy loss
                        if self.config.use_aug_entropy_loss:
                            pg_loss = pg_loss + self.config.aug_entropy_loss_coef * discount_ratio * aug_entropy_loss

                        metrics["actor/kl_prcp_loss"] = - kl_prcp_loss.detach().item()
                        metrics["actor/kl_prcp_coef"] = kl_prcp_coef
                        metrics["actor/kl_prcp_coef_annealing_discount"] = discount_ratio
                        metrics["actor/aug_entropy_loss"] = aug_entropy_loss.detach().item()
                        metrics["actor/aug_entropy_loss_coef"] = self.config.aug_entropy_loss_coef     

                    if self.config.use_ori_entropy_loss:
                        pg_loss = pg_loss + self.config.ori_entropy_loss_coef * discount_ratio * entropy_loss
                        metrics["actor/ori_entropy_loss"] = entropy_loss.detach().item()
                        metrics["actor/ori_entropy_loss_coef"] = self.config.ori_entropy_loss_coef

                    # if adding additional sft loss on correct rollouts
                    if self.config.use_sft_loss:
                        assert correctness_mult_mask is not None, "correctness_mult_mask must be provided when use_sft_loss is True"
                        correctness_mult_mask = torch.tensor(correctness_mult_mask, device=log_probs.device, dtype=log_probs.dtype) # (bsz,) 0.0 or 1.0
                        # token-level mask = response tokens that belong to *correct* samples
                        combined_mask = response_mask * correctness_mult_mask.unsqueeze(1)      # (bsz, response_len)
                        # negative log-probability (NLL) of the chosen tokens
                        sft_loss = VF.masked_mean(-log_probs, combined_mask)                    # scalar
                        sft_coef = self.config.sft_loss_coef # default to 1e-3
                        # add sft loss to pg_loss
                        pg_loss = pg_loss + sft_coef * sft_loss
                        # logging
                        metrics["actor/sft_loss"] = sft_loss.detach().item()
                        metrics["actor/sft_coef"] = sft_coef               

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
