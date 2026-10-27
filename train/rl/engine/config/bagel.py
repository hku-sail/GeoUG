from ml_collections import config_dict
import importlib.util
import os

_BASE_PATH = os.path.join(os.path.dirname(__file__), "base.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_geoug_rl_base_config", _BASE_PATH)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

huggingface_models_root = os.environ.get("MODEL_ROOT", "")


def _base_config():
    """Shared base config for all GeoUG RL engine experiments.

    This flattens the original 11-level inheritance chain into a single base.
    All values are the fully-resolved common state across all experiment configs.
    """
    config = base.get_config()

    # ===== Model =====
    config.model = config_dict.ConfigDict()
    config.model.bagel_model_path = os.environ.get(
        "MODEL_PATH", os.path.join(huggingface_models_root, "BAGEL-7B-MoT"))
    config.model.load_from_ema = True
    config.model.vae_path = os.path.join(config.model.bagel_model_path, "ae.safetensors")
    config.model.max_latent_size = 64
    config.model.max_sequence_length = 1024
    config.model.vae_max_image_size = 1024
    config.model.vae_min_image_size = 512
    config.model.vit_max_image_size = 980
    config.model.vit_min_image_size = 378

    # ===== General =====
    config.mixed_precision = "bf16"
    config.resolution = 512
    config.use_lora = True
    config.lora_rank = 32
    config.lora_alpha = 64
    config.num_epochs = 10
    config.num_checkpoint_limit = 10
    config.save_freq = 20
    config.eval_freq = 10
    config.log_image_freq = 10
    config.max_images_per_log = 8
    config.sanity_check = False
    config.clean_cache = True
    
    config.image_root = None
    config.resume_from = None
    config.logger_type = 'wandb'
    config.per_prompt_stat_tracking = True

    # ===== Wandb =====
    config.wandb = config_dict.ConfigDict()
    config.wandb.project = "alphagrpo"

    # ===== Sampling =====
    config.sample.prefix_think_text = None
    config.sample.num_batches_per_ppo_epoch = 1
    config.sample.eval_num_steps = 40
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 2
    config.sample.per_num_batches_to_record = 2
    config.sample.timestep_shift = 3.0

    config.sample.cfg_text_scale = 4.0
    config.sample.cfg_image_scale = 2.0
    config.sample.cfg_interval = [0.4, 1.0]
    config.sample.cfg_interval_edit = [0.0, 1.0]
    config.sample.cfg_renorm_min = 0.0
    config.sample.cfg_renorm_type = "global"
    config.sample.cfg_renorm_type_edit = 'text_channel'

    config.sample.noise_level = 0.7
    config.sample.global_std = True
    config.sample.kl_reward = 0

    config.sample.think_text_temperature = 1.0
    config.sample.think_text_top_p = 0.8
    config.sample.think_text_top_k = None
    config.sample.eval_think_text_temperature = 0.3
    config.sample.eval_think_text_top_p = 1.0
    config.sample.eval_think_text_top_k = None
    config.sample.repeat_think_text = 1
    config.sample.use_flowcps = False  # use Flow-CPS SDE variant
    config.sample.enable_sde_window = True
    config.sample.random_sde_window = True
    config.sample.sde_window_range = None  # (min_step, max_step); None = (0, num_steps)
    config.sample.stage1_reduction_ratio = 0.5

    # ===== Training =====
    config.train.deepspeed_config = 'scripts/accelerate_configs/deepspeed_zero2.json'
    config.train.use_gradient_checkpointing = True
    config.train.learning_rate = 5e-5
    config.train.adam_weight_decay = 1e-4
    config.train.und_learning_rate = None  # separate lr for understanding modules (None = use main lr)
    config.train.use_liger_kernel = False  # Liger kernel optimization for Qwen2

    config.train.disable_image_grpo = False
    config.train.dataloader_num_workers = 4
    config.train.warmup_steps = 0
    config.train.batch_size = 2
    # num_gpus = training GPUs only (excludes vLLM reward server GPUs)
    #   64-GPU cluster: 8 GPUs for vLLM → 56 training GPUs
    #   32-GPU cluster: 4 GPUs for vLLM → 28 training GPUs
    # ga = total_batch_size * num_image_per_prompt / (num_gpus * train_batch_size)
    config.num_gpus = 56
    config.total_batch_size = 32
    config.train.num_inner_epochs = 1
    config.train.ema = False
    config.train.ema_decay = 0.9999
    config.train.algorithm = 'grpo'  # text algorithm
    config.train.image_algorithm = None  # if None, defaults to algorithm
    config.train.task = 'reflect'
    config.train.ppo_epoch = 1
    config.train.async_reward_fn = True   # if use mllm as reward function, recommended to set to True
    config.train.backward_calls_per_sample = None  # override gradient accumulation calculation
    config.train.shuffle_train_sample = False  # shuffle sample order within each training step

    config.train.recompute_log_prob = False
    config.train.true_ratio = True
    config.train.adv_clip_max = 5
    config.train.image_clip_range = 1e-5
    config.train.text_clip_range = 0.20
    config.train.text_clip_range_high = 0.28
    config.train.text_loss_weight = 1.0

    # use gspo for text
    config.train.text_importance_sampling_level = "sequence"
    config.train.text_loss_level = "token"

    config.train.image_beta = 0.0
    config.train.text_beta = 0.0

    config.train.use_think_text_format_reward = True
    config.train.think_text_format_reward_weight = 0.5
    config.train.think_text_weight = 1.0

    # for self-reflective refinement
    config.train.clamp_no_improve_reward = True
    config.train.mix_t2i_task = False
    config.train.stage1_reward_as_group_mean = False

    config.train.eval_before_train = False
    config.train.reflect_think_text_format = False
    config.train.mask_image_adv_w_negative_think_text = False
    config.train.isolate_image_text_reward = False
    config.train.init_same_noise = False
    config.train.top_entropy_quantile = 1.0

    config.train.sft = 0.0
    config.train.sft_batch_size = 3
    config.train.last_ode = True
    config.train.rollout_correction = False
    config.train.correction_truncated_threshold = 2.0
    config.train.reward_reduce_stage1 = False
    config.train.sample_stage1_reward = False
    config.train.win_lose_reward = False

    # ===== Text Reward =====
    config.text_reward_fn = {
        "think_tag_format": 1.0,
    }

    # Remove pretrained config from base
    if 'pretrained' in config:
        del config['pretrained']

    return config
