# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm
import math


def sde_step(sample, model_output, sigma, sigma_prev, sigma_max, dt, noise_level=0.7, prev_sample=None,
             generator=None):
    # flow-sde
    model_output = model_output.float().to(sample.device)
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    std_dev_t = torch.sqrt(
        sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

    prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + \
                       model_output * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt

    if prev_sample is None:
        noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * noise

    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-dt)) ** 2)) 
                # - torch.log(std_dev_t * torch.sqrt(-dt))
                # - torch.log(torch.sqrt(torch.as_tensor(2 * math.pi)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def sde_step_with_flowcps(sample, model_output, sigma, sigma_prev, sigma_max, dt, noise_level=0.7,
                          prev_sample=None, generator=None):
    # flow-cps
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
    pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
    noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
    prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
        sigma_prev ** 2 - std_dev_t ** 2)

    if prev_sample is None:
        noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * noise

    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    return prev_sample, log_prob, prev_sample_mean, std_dev_t



def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


# from trl
def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class BagelConfig(PretrainedConfig):
    def __init__(
            self,
            visual_gen=True,
            visual_und=True,
            llm_config=None,
            vit_config=None,
            vae_config=None,
            latent_patch_size=2,
            max_latent_size=32,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            interpolate_pos=False,
            timestep_shift=1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

    def to_diff_dict(self) -> dict:
        base = super().to_diff_dict()

        for key in ["llm_config", "vit_config", "vae_config"]:
            val = getattr(self, key, None)
            if val is not None:
                try:
                    base[key] = val.to_dict() if isinstance(val, PretrainedConfig) else dict(val)
                except Exception as e:
                    base[key] = str(val)
        return base


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.config.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        self._head_dtype = torch.float32

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()
        self.progress_bar_config = {}

    @property
    def head_dtype(self):
        return self._head_dtype

    @head_dtype.setter
    def head_dtype(self, dtype: torch.dtype):
        self._head_dtype = dtype

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def set_progress_bar_config(self, **kwargs):
        self.progress_bar_config = kwargs

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.model.gradient_checkpointing = True

    def forward(self, *args, forward_mode='forward', **kwargs):
        return getattr(self, '_' + forward_mode)(*args, **kwargs)

    def _forward(
            self,
            sequence_length: int,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            sample_lens: List[int],
            packed_position_ids: torch.LongTensor,
            nested_attention_masks: List[torch.Tensor] = None,
            split_lens: List[int] = None,
            attn_modes: List[str] = None,
            # for visual understanding
            ce_loss_indexes: Optional[torch.BoolTensor] = None,
            packed_label_ids: Optional[torch.LongTensor] = None,
            packed_vit_tokens: Optional[torch.Tensor] = None,
            packed_vit_token_indexes: Optional[torch.LongTensor] = None,
            packed_vit_position_ids: Optional[torch.LongTensor] = None,
            vit_token_seqlens: Optional[torch.IntTensor] = None,
            # for visual generation
            padded_latent: Optional[torch.Tensor] = None,
            patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
            packed_latent_position_ids: Optional[torch.LongTensor] = None,
            packed_vae_token_indexes: Optional[torch.LongTensor] = None,
            packed_timesteps: Optional[torch.LongTensor] = None,
            mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (
                    1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:,
            None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen and len(mse_loss_indexes):
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean  # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None and len(ce_loss_indexes):
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)

    def _forward_logits(
            self,
            sequence_length: int,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            sample_lens: List[int],
            packed_position_ids: torch.LongTensor,
            nested_attention_masks: List[torch.Tensor] = None,
            split_lens: List[int] = None,
            attn_modes: List[str] = None,
            # for visual understanding
            ce_loss_indexes: Optional[torch.BoolTensor] = None,
            packed_label_ids: Optional[torch.LongTensor] = None,
            packed_vit_tokens: Optional[torch.Tensor] = None,
            packed_vit_token_indexes: Optional[torch.LongTensor] = None,
            packed_vit_position_ids: Optional[torch.LongTensor] = None,
            vit_token_seqlens: Optional[torch.IntTensor] = None,
            # for visual generation
            packed_latent: Optional[torch.Tensor] = None,
            padded_images: Optional[torch.Tensor] = None,
            patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
            packed_latent_position_ids: Optional[torch.LongTensor] = None,
            packed_vae_token_indexes: Optional[torch.LongTensor] = None,
            packed_timesteps: Optional[torch.LongTensor] = None,
            mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)

            # attn_mask = create_mask(
            #     sparse_mask, B=1, H=1, Q_LEN=seqlen, KV_LEN=seqlen,
            #     device=packed_text_embedding.device,
            # )

            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=packed_text_embedding.device, BLOCK_SIZE=128,
                _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            if vit_token_seqlens is not None:
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=packed_vit_tokens,
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                packed_vit_token_embed = self.connector(packed_vit_token_embed)
                vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            if packed_latent is not None:
                packed_timestep_embeds = self.time_embedder(packed_timesteps.to(self.dtype))
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb

                packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        packed_mse_preds = None
        if self.config.visual_gen:
            with torch.autocast('cuda', dtype=self.head_dtype):
                packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes].to(self.head_dtype))

        packed_ce_preds = None
        if ce_loss_indexes is not None:
            with torch.autocast('cuda', dtype=self.head_dtype):
                packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes].to(self.head_dtype))

        return dict(packed_ce_preds=packed_ce_preds, packed_mse_preds=packed_mse_preds)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def _forward_cache_update_text(self, *args, **kwargs):
        return self.forward_cache_update_text(*args, **kwargs)

    def forward_cache_update_text(
            self,
            past_key_values: NaiveCache,
            packed_text_ids: torch.IntTensor,
            packed_text_position_ids: torch.LongTensor,
            text_token_lens: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_key_value_indexes: torch.LongTensor,
            key_values_lens: torch.IntTensor,
            use_flex_attention=False,
            temperature: float = 1.0, # for calculate the logprob
            return_logprobs=False,
            return_logits=False,
            compute_entropy=False,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            use_flex_attention=use_flex_attention,
            **extra_inputs,
        )
        past_key_values = output.past_key_values


        if return_logits:
            # Get logits from model output
            with torch.autocast('cuda', dtype=self.head_dtype):
                logits = self.language_model.lm_head(output.packed_query_sequence.to(self.head_dtype))

            return past_key_values, logits
        elif return_logprobs:
            # Get logits from model output
            with torch.autocast('cuda', dtype=self.head_dtype):
                logits = self.language_model.lm_head(output.packed_query_sequence.to(self.head_dtype))

            logits = logits / temperature

            # Handle packed sequence by splitting according to text_token_lens
            logits_list = torch.split(logits, text_token_lens.tolist(), dim=0)
            input_ids_list = torch.split(packed_text_ids, text_token_lens.tolist(), dim=0)

            # Compute log probabilities for each sequence
            per_token_log_probs = []

            for seq_logits, seq_label_ids in zip(logits_list, input_ids_list):
                # Exclude last logit and first input_id for alignment
                seq_logits = seq_logits[:-1]  # (L-1, V)
                seq_label_ids = seq_label_ids[1:]  # (L-1,)

                # Compute log probabilities
                log_probs = torch.log_softmax(seq_logits.float(), dim=-1, dtype=torch.float32)
                token_log_prob = torch.gather(log_probs, dim=1,
                                              index=seq_label_ids.unsqueeze(1).to(log_probs.device)).squeeze(1)
                per_token_log_probs.append(token_log_prob)
                                
            entropy = None
            if compute_entropy:
                with torch.no_grad():
                    # Compute entropy from logits
                    entropy = entropy_from_logits(logits)
                    entropy_list = []
                    for seq_entropy in torch.split(entropy, text_token_lens.tolist(), dim=0):
                        # Exclude last logit and first input_id for alignment
                        seq_entropy = seq_entropy[:-1]  # (L-1, V)
                        entropy_list.append(seq_entropy)
                entropy = torch.cat(entropy_list)

            return past_key_values, torch.cat(per_token_log_probs), text_token_lens - 1, entropy
        else:
            return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def _forward_cache_update_vit(self, *args, **kwargs):
        return self.forward_cache_update_vit(*args, **kwargs)

    # @torch.no_grad
    def forward_cache_update_vit(
            self,
            past_key_values: NaiveCache,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_vit_tokens: torch.Tensor,
            packed_vit_token_indexes: torch.LongTensor,
            packed_vit_position_ids: torch.LongTensor,
            vit_token_seqlens: torch.IntTensor,
            packed_position_ids: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            packed_indexes: torch.LongTensor,
            packed_key_value_indexes: torch.LongTensor,
            key_values_lens: torch.IntTensor,
            use_flex_attention=False,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens.to(torch.bfloat16),
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            use_flex_attention=use_flex_attention,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        generation_input = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in
                            generation_input.items()}

        return generation_input, newlens, new_rope

    def _forward_cache_update_vae(self, *args, **kwargs):
        return self.forward_cache_update_vae(*args, **kwargs)

    # @torch.no_grad
    def forward_cache_update_vae(
            self,
            vae_model,
            past_key_values: NaiveCache,
            padded_images: torch.Tensor,
            patchified_vae_latent_shapes: List,
            packed_vae_position_ids: torch.LongTensor,
            packed_timesteps: torch.Tensor,
            packed_vae_token_indexes: torch.LongTensor,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_position_ids: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            packed_indexes: torch.LongTensor,
            key_values_lens: torch.IntTensor,
            packed_key_value_indexes: torch.Tensor,
            padded_latent=None,
            use_flex_attention=False,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if padded_latent is None:
            padded_latent = vae_model.encode(padded_images.to(torch.bfloat16))

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            use_flex_attention=use_flex_attention,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, generators=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for idx, ((H, W), curr_kvlen, curr_position_id) in enumerate(zip(image_sizes, curr_kvlens, curr_rope)):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w

            if generators is not None:
                g = generators[idx]
            else:
                g = None

            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2, generator=g)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
            self,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_init_noises: torch.Tensor,
            packed_vae_position_ids: torch.LongTensor,
            packed_vae_token_indexes: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            packed_position_ids: torch.LongTensor,
            packed_indexes: torch.LongTensor,
            past_key_values: NaiveCache,
            key_values_lens: torch.IntTensor,
            packed_key_value_indexes: torch.LongTensor,
            num_timesteps: int = 24,
            timestep_shift: float = 1.0,
            cfg_renorm_min: float = 0.0,
            cfg_renorm_type: str = "global",
            cfg_interval: Optional[Tuple[float, float]] = [0, 1],
            # cfg_text
            cfg_text_scale: float = 1.0,
            cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_text_past_key_values: Optional[NaiveCache] = None,
            cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
            cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            # cfg_img
            cfg_img_scale: float = 1.0,
            cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_img_past_key_values: Optional[NaiveCache] = None,
            cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
            cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            cfg_type: str = "parallel",
            use_flex_attention=False,
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), **self.progress_bar_config):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                use_flex_attention=use_flex_attention,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    def get_timesteps(self, num_timesteps: int, timestep_shift=1.0, device='cpu'):
        timesteps_all = torch.linspace(1, 0, num_timesteps, device=device)
        timesteps_all = timestep_shift * timesteps_all / (1 + (timestep_shift - 1) * timesteps_all)
        return timesteps_all

    @torch.no_grad
    def generate_image_with_sde(
            self,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_init_noises: torch.Tensor,
            packed_vae_position_ids: torch.LongTensor,
            packed_vae_token_indexes: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            packed_position_ids: torch.LongTensor,
            packed_indexes: torch.LongTensor,
            past_key_values: NaiveCache,
            key_values_lens: torch.IntTensor,
            packed_key_value_indexes: torch.LongTensor,
            num_timesteps: int = 24,
            timestep_shift: float = 1.0,
            cfg_renorm_min: float = 0.0,
            cfg_renorm_type: str = "global",
            cfg_interval: Optional[Tuple[float, float]] = [0, 1],
            # cfg_text
            cfg_text_scale: float = 1.0,
            cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_text_past_key_values: Optional[NaiveCache] = None,
            cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
            cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            # cfg_img
            cfg_img_scale: float = 1.0,
            cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_img_past_key_values: Optional[NaiveCache] = None,
            cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
            cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            cfg_type: str = "parallel",

            batch_forward_cfg=True,  # This will be faster
            # SDE specific
            noise_level: float = 0.7,
            deterministic: bool = False,
            sde_window=None,
            return_log_probs: bool = False,
            use_flex_attention=True,

            # CPS
            use_flowcps=False,
    ):      
        x_t = packed_init_noises

        timesteps = self.get_timesteps(num_timesteps, timestep_shift=timestep_shift, device=x_t.device)

        all_log_probs = []
        all_latents = [x_t]

        for i in tqdm(range(num_timesteps - 1), total=num_timesteps - 1, **self.progress_bar_config):
            sigma = timesteps[i]
            sigma_prev = timesteps[i + 1]
            dt = sigma_prev - sigma

            timestep = torch.tensor([sigma] * x_t.shape[0], device=x_t.device)

            if sigma > cfg_interval[0] and sigma <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            v_t = self(
                forward_mode='batch_forward_flow' if batch_forward_cfg else 'forward_flow',
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                use_flex_attention=use_flex_attention,
            )

            # SDE step
            if deterministic or (sde_window is not None and (i >= sde_window[1] or i < sde_window[0])):
                x_t = x_t + v_t.to(x_t.device) * dt
            else:
                sigma_max = timesteps[1]

                if use_flowcps:
                    x_t, log_prob, x_t_mean, std_dev_t = sde_step_with_flowcps(
                        x_t, v_t, sigma, sigma_prev, sigma_max, dt, noise_level,
                    )
                else:
                    x_t, log_prob, x_t_mean, std_dev_t = sde_step(
                        x_t, v_t, sigma, sigma_prev, sigma_max, dt, noise_level,
                    )

                if return_log_probs:
                    # Split log_prob according to VAE sequence lengths and sum over spatial dimensions
                    # Each sample has (packed_seqlens[i] - 2) VAE tokens (excluding start/end tokens)
                    vae_seqlens = packed_seqlens - 2
                    log_prob = torch.stack([log_prob.mean() for log_prob in log_prob.split(vae_seqlens.cpu().tolist(), dim=0)])
                    all_log_probs.append(log_prob)

            all_latents.append(x_t)

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        if return_log_probs:
            return unpacked_latent, all_latents, all_log_probs
        return unpacked_latent

    # @torch.no_grad
    def _forward_flow(
            self,
            x_t: torch.Tensor,
            timestep: torch.LongTensor,
            packed_vae_token_indexes: torch.LongTensor,
            packed_vae_position_ids: torch.LongTensor,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_indexes: torch.LongTensor,
            packed_position_ids: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            key_values_lens: torch.IntTensor,
            past_key_values: NaiveCache,
            packed_key_value_indexes: torch.LongTensor,
            cfg_renorm_min: float = 0.0,
            cfg_renorm_type: str = "global",
            # cfg_text
            cfg_text_scale: float = 1.0,
            cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_text_key_values_lens: Optional[torch.Tensor] = None,
            cfg_text_past_key_values: Optional[NaiveCache] = None,
            cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            # cfg_img
            cfg_img_scale: float = 1.0,
            cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_img_key_values_lens: Optional[torch.Tensor] = None,
            cfg_img_past_key_values: Optional[NaiveCache] = None,
            cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            cfg_type: str = "parallel",
            use_flex_attention=False,
            # cache
            model_pred_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_current: Optional[int] = None,
            model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_text_current: Optional[int] = None,
            model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            use_flex_attention=use_flex_attention,
            **extra_inputs,
        )
        with torch.autocast('cuda', dtype=self.head_dtype):
            v_t = self.llm2vae(output.packed_query_sequence.to(self.head_dtype))
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            with torch.autocast('cuda', dtype=self.head_dtype):
                cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence.to(self.head_dtype))
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            with torch.autocast('cuda', dtype=self.head_dtype):
                cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence.to(self.head_dtype))
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    # assume each image is the same size
                    batch_size = len(packed_seqlens)
                    norm_v_t = torch.cat([torch.norm(_).repeat(_.shape[0]) for _ in v_t.chunk(batch_size)]).unsqueeze(
                        -1)
                    norm_v_t_ = torch.cat([torch.norm(_).repeat(_.shape[0]) for _ in v_t_.chunk(batch_size)]).unsqueeze(
                        -1)
                    # norm_v_t = torch.norm(v_t)
                    # norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def _batch_forward_flow(
            self,
            x_t: torch.Tensor,
            timestep: torch.LongTensor,
            packed_vae_token_indexes: torch.LongTensor,
            packed_vae_position_ids: torch.LongTensor,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            packed_indexes: torch.LongTensor,
            packed_position_ids: torch.LongTensor,
            packed_seqlens: torch.IntTensor,
            key_values_lens: torch.IntTensor,
            past_key_values: NaiveCache,
            packed_key_value_indexes: torch.LongTensor,
            cfg_renorm_min: float = 0.0,
            cfg_renorm_type: str = "global",
            # cfg_text
            cfg_text_scale: float = 1.0,
            cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_text_key_values_lens: Optional[torch.Tensor] = None,
            cfg_text_past_key_values: Optional[NaiveCache] = None,
            cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            # cfg_img
            cfg_img_scale: float = 1.0,
            cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
            cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
            cfg_img_key_values_lens: Optional[torch.Tensor] = None,
            cfg_img_past_key_values: Optional[NaiveCache] = None,
            cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
            cfg_type: str = "parallel",
            use_flex_attention=False,
            # cache
            model_pred_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_current: Optional[int] = None,
            model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_text_current: Optional[int] = None,
            model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
            model_pred_img_current: Optional[int] = None,
    ):
        """
        Optimized version that batches CFG forward passes while maintaining correctness.
        Uses the same sequence for all branches but different past_key_values.
        """
        # Determine which forward passes we need
        need_cfg_text = cfg_text_scale > 1.0
        need_cfg_img = cfg_img_scale > 1.0

        # Prepare the base sequence (same for all forward passes)
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)

        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        # Collect all forward pass configurations
        forward_configs = []

        # Main forward pass
        forward_configs.append({
            'sequence': packed_sequence,
            'query_lens': packed_seqlens,
            'position_ids': packed_position_ids,
            'query_indexes': packed_indexes,
            'past_key_values': past_key_values,
            'key_values_lens': key_values_lens,
            'key_value_indexes': packed_key_value_indexes,
        })

        # CFG text forward pass
        if need_cfg_text:
            forward_configs.append({
                'sequence': packed_sequence,
                'query_lens': packed_seqlens,
                'position_ids': cfg_text_packed_position_ids,
                'query_indexes': cfg_text_packed_query_indexes,
                'past_key_values': cfg_text_past_key_values,
                'key_values_lens': cfg_text_key_values_lens,
                'key_value_indexes': cfg_text_packed_key_value_indexes,
            })

        # CFG img forward pass
        if need_cfg_img:
            forward_configs.append({
                'sequence': packed_sequence,
                'query_lens': packed_seqlens,
                'position_ids': cfg_img_packed_position_ids,
                'query_indexes': cfg_img_packed_query_indexes,
                'past_key_values': cfg_img_past_key_values,
                'key_values_lens': cfg_img_key_values_lens,
                'key_value_indexes': cfg_img_packed_key_value_indexes,
            })

        # Create unified batch inputs
        seq_len = packed_sequence.shape[0]
        num_forward = len(forward_configs)

        unified_sequence = torch.cat([config['sequence'] for config in forward_configs])
        unified_query_lens = torch.cat([config['query_lens'] for config in forward_configs])
        unified_position_ids = torch.cat([config['position_ids'] for config in forward_configs])

        # Adjust query indexes for batched sequence
        unified_query_indexes = []
        unified_key_value_indexes = []
        offset = 0
        for i, config in enumerate(forward_configs):
            unified_query_indexes.append(config['query_indexes'] + offset)
            unified_key_value_indexes.append(config['key_value_indexes'] + offset)
            offset += sum(config['query_lens']) + sum(config['key_values_lens'])

        unified_query_indexes = torch.cat(unified_query_indexes)
        unified_key_value_indexes = torch.cat(unified_key_value_indexes)
        unified_key_values_lens = torch.cat([config['key_values_lens'] for config in forward_configs])
        unified_past_key_values = self._merge_past_key_values_list(
            [config['past_key_values'] for config in forward_configs])

        extra_inputs = {}
        if self.use_moe:
            # indexes in the input sequence. Not the overall sequence(query + kv)
            batched_vae_indexes = []
            batched_text_indexes = []
            offset = 0
            for i in range(num_forward):
                batched_vae_indexes.append(packed_vae_token_indexes + i * seq_len)
                batched_text_indexes.append(packed_text_indexes + i * seq_len)

            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": torch.cat(batched_vae_indexes),
                "packed_text_indexes": torch.cat(batched_text_indexes)
            }

        # Perform single batch forward pass
        batch_output = self.language_model.forward_inference(
            packed_query_sequence=unified_sequence,
            query_lens=unified_query_lens,
            packed_query_position_ids=unified_position_ids,
            packed_query_indexes=unified_query_indexes,
            past_key_values=unified_past_key_values,
            key_values_lens=unified_key_values_lens,
            packed_key_value_indexes=unified_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            use_flex_attention=use_flex_attention,
            **extra_inputs,
        )

        with torch.autocast('cuda', dtype=self.head_dtype):
            # Split the batch output back to individual outputs
            vae_output = self.llm2vae(batch_output.packed_query_sequence.to(self.head_dtype))

        # Split vae_output by sequence length and extract VAE tokens
        outputs = vae_output.split([seq_len] * num_forward)

        v_t = outputs[0][packed_vae_token_indexes]
        cfg_text_v_t = outputs[1][packed_vae_token_indexes] if need_cfg_text else None
        cfg_img_v_t = outputs[2 if need_cfg_text else 1][packed_vae_token_indexes] if need_cfg_img else None

        if need_cfg_text:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if need_cfg_img:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if need_cfg_img:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    # norm_v_t = torch.norm(v_t)
                    # norm_v_t_ = torch.norm(v_t_)
                    batch_size = len(packed_seqlens)
                    norm_v_t = torch.cat([torch.norm(_).repeat(_.shape[0]) for _ in v_t.chunk(batch_size)]).unsqueeze(
                        -1)
                    norm_v_t_ = torch.cat([torch.norm(_).repeat(_.shape[0]) for _ in v_t_.chunk(batch_size)]).unsqueeze(
                        -1)

                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale

        return v_t

    def _merge_past_key_values_list(self, past_key_values_list):
        """
        Create a unified past_key_values by concatenating multiple NaiveCache instances.
        """
        if not past_key_values_list:
            return None

        if len(past_key_values_list) == 1:
            return past_key_values_list[0]

        # Get the first cache as reference
        base_cache = past_key_values_list[0]
        base_num_layers = len(base_cache.key_cache)

        # Assert all caches have the same number of layers
        for i, cache in enumerate(past_key_values_list):
            assert len(cache.key_cache) == base_num_layers, \
                f"Cache {i} has {len(cache.key_cache)} layers, expected {base_num_layers}"
            assert len(cache.value_cache) == base_num_layers, \
                f"Cache {i} has {len(cache.value_cache)} value layers, expected {base_num_layers}"

        # Create new unified cache
        merged_cache = NaiveCache(self.config.llm_config.num_hidden_layers)

        # Concatenate key and value tensors for each layer
        for layer_idx in range(base_num_layers):
            # Collect keys and values from all caches for this layer
            keys_to_concat = []
            values_to_concat = []

            for cache in past_key_values_list:
                if cache.key_cache[layer_idx] is not None:
                    keys_to_concat.append(cache.key_cache[layer_idx])
                if cache.value_cache[layer_idx] is not None:
                    values_to_concat.append(cache.value_cache[layer_idx])

            # Concatenate along sequence dimension (dim=0)
            if keys_to_concat:
                unified_key = torch.cat(keys_to_concat, dim=0)
                merged_cache.key_cache[layer_idx] = unified_key

            if values_to_concat:
                unified_value = torch.cat(values_to_concat, dim=0)
                merged_cache.value_cache[layer_idx] = unified_value

        return merged_cache

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids, prefix_think_token_ids=None):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            if prefix_think_token_ids is not None:
                packed_start_tokens.extend(prefix_think_token_ids)
                packed_query_position_ids.extend(
                    range(curr_position_id + 1, curr_position_id + 1 + len(prefix_think_token_ids)))
                curr_kvlen += len(prefix_think_token_ids)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
            self,
            past_key_values: NaiveCache,
            packed_key_value_indexes: torch.LongTensor,
            key_values_lens: torch.IntTensor,
            packed_start_tokens: torch.LongTensor,
            packed_query_position_ids: torch.LongTensor,
            max_length: int,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            end_token_id: int = None,
            use_flex_attention=False,
            return_kv_cache=False,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        batch_size = len(key_values_lens)

        if not return_kv_cache:
            past_key_values = copy.deepcopy(past_key_values)

        # Initialize finished flag list for each batch item
        finished = [False] * batch_size

        while step < max_length:
            if batch_size == curr_tokens.shape[0]:
                generated_sequence.append(curr_tokens)
                packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
                query_lens = torch.ones_like(curr_tokens)
                packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                    0, len(key_values_lens),
                    device=key_values_lens.device,
                    dtype=key_values_lens.dtype
                )
                uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
                for i in range(len(uppacked)):
                    uppacked[i] += i
                packed_key_value_indexes = torch.cat(uppacked, dim=0)
            else:
                # multiple tokens for each sample.
                for split_curr_tokens in curr_tokens.view(batch_size, -1).permute(1, 0):
                    generated_sequence.append(split_curr_tokens)
                packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
                query_lens = torch.as_tensor([curr_tokens.shape[0] // batch_size for _ in range(batch_size)],
                                             device=key_values_lens.device, dtype=key_values_lens.dtype)

                packed_query_indexes = (torch.cumsum(key_values_lens, dim=0).unsqueeze(-1) + torch.arange(
                    0, curr_tokens.shape[0],
                    device=key_values_lens.device,
                    dtype=key_values_lens.dtype
                ).view(batch_size, -1)).view(-1)

                uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
                for i in range(len(uppacked)):
                    uppacked[i] += i * query_lens[0]
                packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                use_flex_attention=use_flex_attention,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence

            if batch_size < packed_query_sequence.shape[0]:
                # select the last token
                packed_query_sequence = packed_query_sequence.view(
                    batch_size, -1, packed_query_sequence.shape[-1])[:, -1]
                
            with torch.autocast('cuda', dtype=self.head_dtype):
                pred_logits = self.language_model.lm_head(packed_query_sequence.to(self.head_dtype))

                if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
                    raise ValueError(
                        f"Model output contains NaNs/Infs before sampling. "
                        f"head_dtype={self.head_dtype}, "
                        f"max={pred_logits.max().item()}, min={pred_logits.min().item()}"
                    )
                
                if do_sample:
                    pred_logits = pred_logits / temperature
                    if top_k is not None and top_k > 0:
                        top_k = min(top_k, pred_logits.size(-1))
                        indices_to_remove = pred_logits < torch.topk(pred_logits, top_k)[0][..., -1, None]
                        pred_logits[indices_to_remove] = -float('Inf')

                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        pred_logits[indices_to_remove] = -float('Inf')

                    probs = nn.functional.softmax(pred_logits, dim=-1)
                    curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    curr_tokens = torch.argmax(pred_logits, dim=-1)

            # Update curr_tokens: keep the same token for finished batch items
            for i in range(batch_size):
                if not finished[i]:
                    # Check if this batch item just finished
                    if end_token_id is not None and curr_tokens[i] == end_token_id:
                        finished[i] = True

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                new_token_indexes = torch.as_tensor([uppacked[i][-1] + k + 1 for k in range(query_lens[0])],
                                                    device=uppacked[i].device)
                uppacked[i] = torch.cat(
                    [uppacked[i], new_token_indexes], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + query_lens[0]
            packed_query_position_ids = packed_query_position_ids.view(batch_size, -1)[:, -1].view(-1) + query_lens[0]
            step += 1

            # Early stopping: break if all batch items are finished
            if all(finished):
                break

        output_device = generated_sequence[0].device
        outputs = torch.stack([i.to(output_device) for i in generated_sequence], dim=0)
        if return_kv_cache:
            return outputs, past_key_values
        else:
            return outputs

    # for evaluation
    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            new_token_ids,
            image_transform,
            images,
            prompt,
            max_length: int,
            do_sample: bool = False,
            temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:, 0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output