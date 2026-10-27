# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
import wandb
import yaml
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.sft.fsdp_utils import (
    FSDPCheckpoint,
    FSDPConfig,
    fsdp_ema_setup,
    fsdp_ema_update,
    fsdp_wrapper,
    grad_checkpoint_check_fn,
)
from train.sft.model_utils import color_logger, print_model_params
from train.sft.train_utils import (
    append_loss_record,
    create_logger,
    dump_data_indexes,
    get_latest_ckpt,
)


@torch.no_grad()
def seed_new_token_rows(
    model,
    tokenizer,
    num_new_tokens,
    base_tokenizer_path,
    logger,
):
    """Seed newly registered token rows from their original token pieces."""
    if num_new_tokens <= 0:
        return
    base = Qwen2Tokenizer.from_pretrained(base_tokenizer_path)
    embedding = model.language_model.model.embed_tokens.weight
    head = model.language_model.lm_head.weight
    for token_id in range(len(tokenizer) - num_new_tokens, len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        pieces = base.encode(token)
        assert pieces and max(pieces) < len(base), (
            f"{token} tokenises into new ids"
        )
        for weight in (embedding, head):
            weight[token_id] = (
                weight[pieces].float().mean(0).to(weight.dtype)
            )
        logger.info(
            "seeded %s (id %s) from %s",
            token,
            token_id,
            base.convert_ids_to_tokens(pieces),
        )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={
            "help": "Path or Hugging Face repo ID of the language model."
        },
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm in attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings."},
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Decoder-layer class name."},
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint."},
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the vision transformer."},
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size per side."},
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size covered by each latent patch."},
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Vision-transformer patch size."},
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of vision patches per side."},
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Connector MLP activation."},
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings."},
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Vision-transformer hidden layer to use."},
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Use RoPE in the vision transformer."},
    )
    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Text-conditioning dropout probability."},
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "VAE-conditioning dropout probability."},
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Vision-conditioning dropout probability."},
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "Dataset groups and preprocessing configuration."},
    )
    prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "Batches preloaded by each DataLoader worker."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader workers."},
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens in one raw sample."},
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Maximum tokens in a packed batch."},
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "Packed length below which overflow is preferred."},
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum overflow-buffer size."},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Dataset sampling seed."},
    )


@dataclass
class TrainingArguments:
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train the image-generation branch."},
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train the image-understanding branch."},
    )
    results_dir: str = field(
        default="results",
        metadata={"help": "Directory for logs and metrics."},
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Directory for model checkpoints."},
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Weights & Biases resume identifier."},
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "Weights & Biases resume mode."},
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run Weights & Biases offline."},
    )
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed."},
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Resume the latest checkpoint automatically."},
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint path to resume from."},
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load model weights without training state."},
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "Load EMA weights when loading model only."},
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Initialize from a Hugging Face model directory."},
    )
    seed_new_tokens: bool = field(
        default=True,
        metadata={"help": "Seed rows for newly registered special tokens."},
    )
    log_every: int = field(
        default=10,
        metadata={"help": "Logging interval in steps."},
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Checkpoint interval in steps."},
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total optimizer steps."},
    )
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Learning-rate warmup steps."},
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Learning-rate schedule: constant or cosine."},
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate."},
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum cosine-schedule learning rate."},
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW beta1."},
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW beta2."},
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW epsilon."},
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "EMA decay."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient-clipping threshold."},
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Diffusion timestep shift."},
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Image MSE loss weight."},
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Language cross-entropy loss weight."},
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Use per-token CE weights."},
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Target packed-batch token count."},
    )
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of FSDP replica groups."},
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of FSDP shards."},
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy."},
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward-prefetch mode."},
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Offload FSDP parameters to CPU."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze language-model weights."},
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Freeze vision-transformer weights."},
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Freeze VAE weights."},
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze understanding connector layers."},
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Copy initial weights to MoE experts."},
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Use FLEX sequence packing."},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Use a reduced model for debugging."},
    )
    enable_masked_mse: bool = field(
        default=True,
        metadata={"help": "Apply the optional MSE mask."},
    )
    debug_norm_threshold: float = field(
        default=0.0,
        metadata={"help": "Dump batch indexes above this gradient norm."},
    )
    debug_dump_step_start: int = field(
        default=-1,
        metadata={"help": "First step in the index-dump window."},
    )
    debug_dump_step_end: int = field(
        default=-1,
        metadata={"help": "Last step in the index-dump window."},
    )
    profile_step_start: int = field(
        default=-1,
        metadata={"help": "First profiler step; -1 disables profiling."},
    )
    profile_num_steps: int = field(
        default=3,
        metadata={"help": "Number of active profiler steps."},
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=(
                f"{training_args.wandb_name}-run"
                f"{training_args.wandb_runid}"
            ),
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = (
                training_args.finetune_from_ema
                if resume_model_only
                else False
            )
        else:
            resume_model_only = False
            finetune_from_ema = False
        logger.info(f"resume_from: {resume_from}")
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = (
            training_args.finetune_from_ema if resume_model_only else False
        )

    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_args.model_path, "llm_config.json")
        )
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    if training_args.debug:
        os.environ["DEBUG"] = "True"
        logger.warning("debug mode")
        llm_config.num_hidden_layers = 2

    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(
            model_args.llm_path,
            config=llm_config,
        )
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(model_args.model_path, "vit_config.json")
            )
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers += 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(
                model_args.vit_path,
                config=vit_config,
            )

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=(
                os.path.join(model_args.model_path, "ae.safetensors")
                if training_args.finetune_from_hf
                else model_args.vae_path
            )
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model,
        vit_model if training_args.visual_und else None,
        config,
    )
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config
        )

    tokenizer_path = (
        model_args.model_path
        if training_args.finetune_from_hf
        else model_args.llm_path
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if len(tokenizer) > model.language_model.config.vocab_size:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    logger.info(
        f"tokenizer: {len(tokenizer)} tokens "
        f"({num_new_tokens} newly registered), embedding rows: "
        f"{model.language_model.config.vocab_size}"
    )

    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    print_model_params(model)

    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from,
        logger,
        model,
        ema_model,
        resume_from_ema=finetune_from_ema,
    )
    if training_args.seed_new_tokens:
        for current_model in (model, ema_model):
            seed_new_token_rows(
                current_model,
                tokenizer,
                num_new_tokens,
                tokenizer_path,
                logger,
            )
    else:
        logger.info(
            "seed_new_tokens=False: keeping checkpoint rows for "
            f"{num_new_tokens} registered tokens"
        )

    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=grad_checkpoint_check_fn,
    )
    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
        )
    else:
        raise ValueError(
            f"Unsupported lr_scheduler: {training_args.lr_scheduler}"
        )

    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = (
            FSDPCheckpoint.try_load_train_state(
                resume_from,
                optimizer,
                scheduler,
                fsdp_config,
            )
        )

    with open(data_args.dataset_config_file) as stream:
        dataset_meta = yaml.safe_load(stream)
    color_logger(f"dataset_meta:\n{json.dumps(dataset_meta, indent=2)}")
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = (
            model_args.vit_max_num_patch_per_side
        )
    if training_args.visual_gen:
        dataset_config.vae_image_downsample = (
            model_args.latent_patch_size * vae_config.downsample
        )
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = (
            model_args.text_cond_dropout_prob
        )
        dataset_config.vae_cond_dropout_prob = (
            model_args.vae_cond_dropout_prob
        )
        dataset_config.vit_cond_dropout_prob = (
            model_args.vit_cond_dropout_prob
        )

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    start_time = time()
    profiler = None
    logger.info(
        f"Training for {training_args.total_steps} steps, "
        f"starting at {train_step}..."
    )
    for curr_step, data in enumerate(train_loader, start=train_step):
        if (
            profiler is None
            and curr_step == training_args.profile_step_start
            and dist.get_rank() == 0
        ):
            trace_path = os.path.join(
                training_args.results_dir,
                f"trace_step{curr_step}.json.gz",
            )
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=1,
                    active=training_args.profile_num_steps,
                ),
                on_trace_ready=lambda profile: profile.export_chrome_trace(
                    trace_path
                ),
                record_shapes=True,
            )
            profiler.start()
            logger.info(
                f"[profile] recording {training_args.profile_num_steps} "
                f"steps -> {trace_path}"
            )

        data = data.cuda(device).to_dict()
        data_indexes = data.pop("batch_data_indexes", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)
        skip = torch.tensor(
            1.0 if "padded_images" not in data else 0.0,
            device=device,
        )
        dist.all_reduce(skip, op=dist.ReduceOp.MAX)
        if skip.item() > 0 and training_args.visual_gen:
            logger.error(
                f"Rank {dist.get_rank()} has no padded_images; "
                "skipping this batch"
            )
            continue

        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=torch.bfloat16,
        ):
            if training_args.visual_gen:
                with torch.no_grad():
                    data["padded_latent"] = vae_model.encode(
                        data.pop("padded_images")
                    )
            loss_dict = fsdp_model(**data)

        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(
                len(data["ce_loss_indexes"]),
                device=device,
            )
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(
                    total_ce_loss_weights,
                    op=dist.ReduceOp.SUM,
                )
                ce = (
                    ce.sum()
                    * dist.get_world_size()
                    / total_ce_loss_weights
                )
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            if torch.isnan(mse).any():
                raise RuntimeError(
                    "Training terminated due to NaN loss in "
                    f"rank {dist.get_rank()}; samples: {data_indexes}"
                )
            mse = mse.mean(dim=-1)
            total_mse_tokens = torch.tensor(
                len(data["mse_loss_indexes"]),
                device=device,
            )
            mask_tensor = data.get("mask_tensor")
            if mask_tensor is not None and training_args.enable_masked_mse:
                mse = mse * mask_tensor.detach()
                total_mse_tokens = torch.sum(mask_tensor).detach()
                if total_mse_tokens == 0:
                    total_mse_tokens = dist.get_world_size()
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(
            training_args.max_grad_norm
        )
        norm_val = total_norm.item()
        in_window = (
            training_args.debug_dump_step_start >= 0
            and training_args.debug_dump_step_start
            <= curr_step
            <= training_args.debug_dump_step_end
        )
        norm_spike = (
            training_args.debug_norm_threshold > 0
            and norm_val > training_args.debug_norm_threshold
        )
        if in_window or norm_spike:
            if norm_spike:
                logger.warning(
                    f"[diag] step={curr_step} rank={dist.get_rank()} "
                    f"total_norm={norm_val:.2f} > threshold "
                    f"{training_args.debug_norm_threshold}; "
                    "dumping data indexes"
                )
            dump_data_indexes(
                dump_dir=os.path.join(
                    training_args.checkpoint_dir,
                    "debug_data_indexes",
                ),
                step=curr_step,
                rank=dist.get_rank(),
                total_norm=norm_val,
                ce=loss_dict["ce"].item(),
                batch_data_indexes=data_indexes,
                reason="norm_spike" if norm_spike else "window",
            )

        optimizer.step()
        scheduler.step()
        fsdp_ema_update(
            ema_model,
            fsdp_model,
            decay=training_args.ema,
        )

        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(
                len(data["sample_lens"]),
                device=device,
            )
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = (
                training_args.log_every / (end_time - start_time)
            )
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss

            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["total_mse_tokens"] = total_mse_tokens.item()
            wandb_log["total_ce_tokens"] = total_ce_tokens.item()
            wandb_log["total_norm"] = total_norm.item()
            wandb_log["total_samples"] = total_samples.item()
            mem_allocated = torch.tensor(
                torch.cuda.max_memory_allocated() / 1024**2,
                device=device,
            )
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log["mem_allocated"] = mem_allocated
            mem_cache = torch.tensor(
                torch.cuda.max_memory_reserved() / 1024**2,
                device=device,
            )
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log["mem_cache"] = mem_cache
            message += (
                f"lr: {wandb_log['lr']:.3e}, "
                f"total_norm: {wandb_log['total_norm']:.4f}, "
            )
            logger.info(message)

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
                loss_record = {
                    "step": curr_step,
                    "steps_per_sec": steps_per_sec,
                }
                for key, value in wandb_log.items():
                    loss_record[key] = (
                        value.item() if torch.is_tensor(value) else value
                    )
                append_loss_record(
                    training_args.results_dir,
                    loss_record,
                )
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            dataset_status = data_status.setdefault(
                item["dataset_name"],
                {},
            )
            dataset_status[item["worker_id"]] = item["data_indexes"]

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            gather_list = (
                [None] * dist.get_world_size()
                if dist.get_rank() == 0
                else None
            )
            dist.gather_object(data_status, gather_list, dst=0)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
            )

        if profiler is not None:
            profiler.step()
            if (
                curr_step
                >= training_args.profile_step_start
                + training_args.profile_num_steps
            ):
                profiler.stop()
                profiler = None
                logger.info(f"[profile] trace written to {trace_path}")

        if curr_step + 1 >= training_args.total_steps:
            break

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
