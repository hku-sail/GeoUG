# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image

from accelerate.utils import FullyShardedDataParallelPlugin, gather_object
from accelerate.logging import get_logger

logger = get_logger(__name__)


def is_main_process():
    return dist.get_rank() == 0


_global_profilers: dict[str, "TimingProfiler"] = {}


def get_timing_profiler(name: str = "default") -> "TimingProfiler":
    """Get (or create) a global TimingProfiler by name."""
    if name not in _global_profilers:
        _global_profilers[name] = TimingProfiler()
    return _global_profilers[name]


class TimingProfiler:
    """
    A profiler class to record execution times of code blocks.

    Example:
    ```python
    from .utils import TimingProfiler

    profiler = TimingProfiler()

    with profiler("data_loading"):
        # Code for data loading
        time.sleep(0.1)

    with profiler("data_loading"):
        # Code for data loading again
        time.sleep(0.2)

    profiler.start("model_forward")
    # Code for model forward pass
    time.sleep(0.3)
    profiler.end("model_forward")

    # At the end of a loop or epoch
    timing_stats = profiler.pop_records()
    # timing_stats will be {'data_loading': 0.15, 'model_forward': 0.3}
    print(timing_stats)
    ```
    """

    def __init__(self):
        self.records = defaultdict(list)
        self._start_times = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        self.start(name)
        yield
        self.end(name)

    def start(self, name: str):
        """Starts timing for a given name."""
        self._start_times[name] = time.perf_counter()

    def end(self, name: str):
        """Stops timing for a given name and records the duration."""
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started or has already ended.")
        end_time = time.perf_counter()
        duration = end_time - self._start_times.pop(name)
        self.records[name].append(duration)

    def pop_records(self, name=None, to_tensors='pt', device='cuda') -> dict[str, float]:
        """
        Pops all recorded timings, calculates the mean for each named record,
        and returns a dictionary with the results. The internal records are cleared.
        """
        if name is None:
            results = {
                f"Timing/{name}": sum(durations) / len(durations)
                for name, durations in self.records.items()
            }
            self.records.clear()

        else:
            results = {
                f"Timing/{name}": sum(self.records[name]) / len(self.records[name])
            }
            self.records.pop(name)

        if to_tensors == 'pt':
            results = {k: torch.as_tensor(v, dtype=torch.float, device=device) for k, v in results.items()}
        elif to_tensors == 'numpy':
            results = {k: np.asarray(v, dtype=np.float32) for k, v in results.items()}

        return results
    

def apply_liger_kernel_to_qwen2_navit(
    rope: bool = True,
    cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = False,
    model = None,
) -> None:
    
    from types import MethodType

    import transformers

    from packaging import version

    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
    from liger_kernel.transformers.model.qwen2 import lce_forward_deprecated as qwen2_lce_forward_deprecated
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module, _patch_swiglu_module

    transformer_version = version.parse(transformers.__version__)

    SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
    TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            print(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            if model is not None:
                model.forward = MethodType(qwen2_lce_forward, model)
            else:
                modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward
        else:  # if version < 4.46.1
            print(TRANSFORMER_DEPRECATION_WARNING)
            if model is not None:
                model.forward = MethodType(qwen2_lce_forward_deprecated, model)
            else:
                modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward_deprecated

    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
            _patch_rms_norm_module(base_model.norm_moe_gen)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)

                _patch_swiglu_module(decoder_layer.mlp_moe_gen, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

                _patch_rms_norm_module(decoder_layer.input_layernorm_moe_gen)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm_moe_gen)


# ====== Distributed Helpers ======

def scatter_data(data):
    """Broadcast data from rank 0 and scatter to all processes."""
    if isinstance(data, list):
        dist.broadcast_object_list(data, src=0)
    else:
        device = data.device
        data = data.cuda()
        dist.broadcast(data, src=0)
        data = data.to(device)

    num_processes = dist.get_world_size()
    return data[dist.get_rank()::num_processes]


def reduction_prompt_num(prompts, prompt_metadata, stage1_reduction_ratio):
    """Reduce prompts for stage1 by deduplicating and distributing across GPUs."""
    gathered_prompts = gather_object(prompts)
    gathered_prompt_metadata = gather_object(prompt_metadata)

    unique_prompt_metadata = dict()
    for prompt, metadata in zip(gathered_prompts, gathered_prompt_metadata):
        unique_prompt_metadata[prompt] = metadata
    unique_prompts = list(unique_prompt_metadata.keys())
    unique_prompt_metadata = list(unique_prompt_metadata.values())

    num_image_per_prompt = len(gathered_prompts) // len(unique_prompts)
    stage1_num_image_per_prompt = dist.get_world_size() // len(unique_prompts) + int(dist.get_world_size() % len(unique_prompts) > 0)

    all_stage1_prompts = unique_prompts * stage1_num_image_per_prompt
    all_stage1_prompt_metadata = unique_prompt_metadata * stage1_num_image_per_prompt

    return all_stage1_prompts, all_stage1_prompt_metadata


def create_generator(prompts, base_seed):
    """Create per-prompt deterministic generators for reproducible sampling."""
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2 ** 31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def sample_sde_window(config):
    """Sample SDE window for training timesteps."""
    num_train_timesteps = max(int(config.sample.num_steps * config.train.timestep_fraction), 1)

    if config.train.disable_image_grpo:
        sde_window = (0, -1)
    elif config.sample.random_sde_window:
        min_step, max_step = config.sample.sde_window_range or (0, config.sample.num_steps)
        latest_start = max_step - num_train_timesteps
        start_index = random.randint(min_step, latest_start) if latest_start > min_step else min_step
        sde_window = (start_index, start_index + num_train_timesteps)
    else:
        sde_window = (0, num_train_timesteps)
    return sde_window


def is_deepspeed_enabled(accelerator):
    """Check if DeepSpeed is enabled."""
    return accelerator.distributed_type.value == "DEEPSPEED"


def is_deepspeed_zero_stage_3(accelerator):
    """Check if DeepSpeed ZeRO Stage 3 is enabled."""
    if not is_deepspeed_enabled(accelerator):
        return False
    try:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
        return zero_stage == 3
    except Exception:
        return False


def get_deepspeed_info(accelerator):
    """Get DeepSpeed configuration info string."""
    if not is_deepspeed_enabled(accelerator):
        return "DeepSpeed: Disabled"
    try:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        zero_stage = ds_config.get("zero_optimization", {}).get("stage", "Unknown")
        offload_optimizer = ds_config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none")
        offload_param = ds_config.get("zero_optimization", {}).get("offload_param", {}).get("device", "none")
        return f"DeepSpeed: Enabled (Zero Stage: {zero_stage}, Offload Optimizer: {offload_optimizer}, Offload Param: {offload_param})"
    except Exception as e:
        return f"DeepSpeed: Enabled (Could not get config details: {e})"


def get_fsdp_plugin_from_json(config_path):
    """Load FSDP plugin from JSON config file."""
    with open(config_path, 'r') as f:
        fsdp_config = json.load(f)

    if fsdp_config.get('mixed_precision_policy', None) is not None:
        mixed_precision_policy = fsdp_config.get('mixed_precision_policy', None)
        str2dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        fsdp_config['mixed_precision_policy'] = torch.distributed.fsdp.MixedPrecisionPolicy({k: str2dtype[v] for k, v in mixed_precision_policy.items()})

    fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_config)
    return fsdp_plugin


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Calculate the proportion of unique prompts whose reward standard deviation is zero."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


def get_high_entropy_mask(entropies: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

    Args:
        entropies (`torch.Tensor`):
            Tensor of shape (batch_size x seq_len) with per-token entropy values.
        threshold (`float`):
            Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

    Returns:
        `torch.Tensor`:
            Boolean mask of shape (batch_size x seq_len), where `True` indicates tokens with entropy >= threshold
            and `False` otherwise.
    """
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, entropies.detach().cpu())
    gathered = torch.cat(gathered)

    if gathered.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.bool)

    entropy_threshold = torch.quantile(gathered, threshold)
    entropy_mask = entropies >= entropy_threshold
    return entropy_mask



def save_ckpt(save_dir, bagel_model, global_step, accelerator, ema, config):
    """Save checkpoint with DeepSpeed support"""
    os.makedirs(save_dir, exist_ok=True)

    # Use accelerator's save_state for DeepSpeed compatibility
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{global_step}")

    if accelerator.is_main_process:
        print(f'Saving checkpoint to {checkpoint_dir}.')

    model_trainable_parameters = list(filter(lambda p: p.requires_grad, bagel_model.parameters()))

    if accelerator.distributed_type.value in ["DEEPSPEED", "FSDP"]:
        # For DeepSpeed and FSDP, use accelerator.save_state to handle distributed saving
        if ema is not None:
            ema.copy_ema_to(model_trainable_parameters, store_temp=True)
            accelerator.save_state(checkpoint_dir)
            ema.copy_temp_to(model_trainable_parameters)
        else:
            accelerator.save_state(checkpoint_dir)

        # Save additional model files if needed
        # Try to save model in HuggingFace format for easier loading later
        if ema is not None:
            ema.copy_ema_to(model_trainable_parameters, store_temp=True)
            unwrapped_model = accelerator.unwrap_model(bagel_model)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, "hf_model"))
            ema.copy_temp_to(model_trainable_parameters)
        else:
            unwrapped_model = accelerator.unwrap_model(bagel_model)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, "hf_model"))
    else:
        # Original saving logic for non-DeepSpeed training
        if accelerator.is_main_process:
            if ema is not None:
                ema.copy_ema_to(model_trainable_parameters, store_temp=True)
                unwrapped_model = accelerator.unwrap_model(bagel_model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                ema.copy_temp_to(model_trainable_parameters)
            else:
                unwrapped_model = accelerator.unwrap_model(bagel_model)
                unwrapped_model.save_pretrained(checkpoint_dir)

    # Save config
    if accelerator.is_main_process:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)


def sanitize_config(cfg):
    """Flatten nested config dict for TensorBoard logging (convert unsupported types to string)."""
    flat = {}
    for k, v in cfg.items():
        # Convert dataclass objects to dict
        if hasattr(v, "__dict__"):
            v = vars(v)
        # Convert list, tuple, or dict to string
        if isinstance(v, (list, tuple, dict)):
            v = str(v)
        # Convert other unsupported types to string
        if not isinstance(v, (int, float, str, bool, torch.Tensor)):
            v = str(v)
        flat[k] = v
    return flat


# ====== Image Logging ======

def concat_images_list(image_list1, image_list2):
    concated_images = []
    for image1, image2 in zip(image_list1, image_list2):
        height = image1.size[1] + image2.size[1]
        width = max(image1.size[0], image2.size[0])
        new_im = Image.new("RGB", (width, height))
        new_im.paste(image1, (0, 0))
        new_im.paste(image2, (0, image1.size[1]))
        concated_images.append(new_im)
    return concated_images


def log_images_to_tracker(images, prompts, rewards, accelerator, global_step, config, max_images=4, prefix="train"):
    import wandb

    if not accelerator.is_main_process or not images:
        return

    n = min(len(images), max_images)
    cols = min(4, n)
    rows = -(-n // cols)  # ceiling division
    img_w, img_h = images[0].size

    grid_image = Image.new('RGB', (cols * img_w, rows * img_h), color='white')
    for i, image in enumerate(images[:n]):
        grid_image.paste(image, ((i % cols) * img_w, (i // cols) * img_h))

    caption_parts = []
    for i in range(n):
        caption_parts.append(f"Image {i + 1}: {prompts[i]}")
        if isinstance(rewards, dict):
            for key, vals in rewards.items():
                vals_np = vals.cpu().numpy() if hasattr(vals, 'cpu') else vals
                if i < len(vals_np):
                    caption_parts.append(f"  {key}: {vals_np[i]:.4f}")
        caption_parts.append("")
    caption = "\n".join(caption_parts).strip()

    if config.logger_type == "wandb":
        wandb.log({f"{prefix}/sample_images": wandb.Image(grid_image, caption=caption)}, step=global_step)
    else:
        import torchvision.transforms as transforms
        grid_tensor = transforms.ToTensor()(grid_image)

        tracker = accelerator.get_tracker("tensorboard")
        writer = getattr(tracker, 'writer', None) or getattr(tracker, '_tracker', None)
        if writer is None:
            accelerator.log({f"{prefix}/sample_images": grid_tensor}, step=global_step)
            accelerator.log({f"{prefix}/sample_captions": caption}, step=global_step)
        else:
            writer.add_image(f"{prefix}/sample_images", grid_tensor, global_step=global_step)
            writer.add_text(f"{prefix}/sample_captions", caption, global_step=global_step)

    logger.info(f"Logged grid image ({rows}x{cols}) with {n} images at step {global_step}")

