"""BAGEL depth-generation adapter used by the public evaluator."""

import copy
import gc
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional
import yaml
from accelerate import init_empty_weights
from loguru import logger
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from data.transforms_depth import build_depth_transform, decode_depth_wrap
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.qwen2 import Qwen2Tokenizer


def _move_to_device(inputs, device):
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device, non_blocking=True)
    return inputs


def _scaled_size(width: int, height: int, scale: float):
    def divisible(value: float):
        return max(16, int(round(value / 16) * 16))

    return divisible(width * scale), divisible(height * scale)


class BagelDepthModel:
    def __init__(
        self,
        model_path: str,
        transform_config: str,
        device: torch.device,
        is_ema: bool = True,
        max_latent_size: int = 64,
    ) -> None:
        model_root = Path(model_path).expanduser()
        llm_config = Qwen2Config.from_json_file(
            model_root / "llm_config.json"
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(
            model_root / "vit_config.json"
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(
            local_path=model_root / "ae.safetensors"
        )

        with Path(transform_config).open(encoding="utf-8") as stream:
            transform = yaml.safe_load(stream)
        self.vae_transform = ImageTransform(
            **transform["image_transform_args"]
        )
        self.vit_transform = ImageTransform(
            **transform["vit_image_transform_args"]
        )
        self.depth_transform = (
            build_depth_transform(transform["depth_transform_args"])
            if "depth_transform_args" in transform
            else None
        )

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=max_latent_size,
        )
        # Avoid allocating and randomly initializing a full FP32 copy of the
        # 14B model before loading its BF16 checkpoint. Buffers stay on CPU;
        # parameters are materialized directly from the safetensors tensors.
        with init_empty_weights():
            model = Bagel(
                Qwen2ForCausalLM(llm_config),
                SiglipVisionModel(vit_config),
                config,
            )
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config
            )
        tokenizer = Qwen2Tokenizer.from_pretrained(model_root)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)

        weights = model_root / (
            "ema.safetensors" if is_ema else "model.safetensors"
        )
        state_dict = load_file(weights, device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {weights}: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        logger.info("Loaded {} with all tensors matched", weights)
        del state_dict

        self.device = device
        self.gen_model = model.to(device, dtype=torch.bfloat16).eval()
        self.vae_model = vae_model.to(device).eval()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def inference(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        num_timesteps: int = 50,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        include_vit_incontext: bool = True,
        max_image_size: int = 1024,
        min_image_size: int = 512,
    ) -> torch.Tensor:
        width, height = images[0].size
        scale = min(max_image_size / max(width, height), 1.0)
        scale = max(scale, min_image_size / min(width, height))
        width, height = _scaled_size(width, height, scale)
        if max(width, height) > max_image_size:
            width, height = _scaled_size(
                width, height, max_image_size / max(width, height)
            )

        layer_count = self.gen_model.config.llm_config.num_hidden_layers
        cache = NaiveCache(layer_count)
        lengths, rope = [0], [0]
        for image in images:
            inputs, lengths, rope = self.gen_model.prepare_vae_images(
                curr_kvlens=lengths,
                curr_rope=rope,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            inputs = _move_to_device(inputs, self.device)
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.bfloat16
            ):
                cache = self.gen_model.forward_cache_update_vae(
                    self.vae_model, cache, **inputs
                )
            if include_vit_incontext:
                inputs, lengths, rope = self.gen_model.prepare_vit_images(
                    curr_kvlens=lengths,
                    curr_rope=rope,
                    images=[image],
                    transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
                inputs = _move_to_device(inputs, self.device)
                with torch.autocast(
                    "cuda", enabled=True, dtype=torch.bfloat16
                ):
                    cache = self.gen_model.forward_cache_update_vit(
                        cache, **inputs
                    )

        text_cfg_cache = copy.deepcopy(cache)
        text_cfg_inputs = self.gen_model.prepare_vae_latent_cfg(
            curr_kvlens=lengths,
            curr_rope=rope,
            image_sizes=[(height, width)],
        )
        text_cfg_inputs = _move_to_device(text_cfg_inputs, self.device)

        image_cfg_cache = NaiveCache(layer_count)
        image_cfg_lengths, image_cfg_rope = [0], [0]
        image_cfg_inputs, image_cfg_lengths, image_cfg_rope = (
            self.gen_model.prepare_prompts(
                curr_kvlens=image_cfg_lengths,
                curr_rope=image_cfg_rope,
                prompts=[prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
        )
        image_cfg_inputs = _move_to_device(image_cfg_inputs, self.device)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            image_cfg_cache = self.gen_model.forward_cache_update_text(
                image_cfg_cache, **image_cfg_inputs
            )
        image_cfg_inputs = self.gen_model.prepare_vae_latent_cfg(
            curr_kvlens=image_cfg_lengths,
            curr_rope=image_cfg_rope,
            image_sizes=[(height, width)],
        )
        image_cfg_inputs = _move_to_device(image_cfg_inputs, self.device)

        inputs, lengths, rope = self.gen_model.prepare_prompts(
            curr_kvlens=lengths,
            curr_rope=rope,
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        inputs = _move_to_device(inputs, self.device)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cache = self.gen_model.forward_cache_update_text(cache, **inputs)
        inputs = self.gen_model.prepare_vae_latent(
            curr_kvlens=lengths,
            curr_rope=rope,
            image_sizes=[(height, width)],
            new_token_ids=self.new_token_ids,
        )
        inputs = _move_to_device(inputs, self.device)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            latent = self.gen_model.generate_image(
                past_key_values=cache,
                cfg_text_past_key_values=text_cfg_cache,
                cfg_img_past_key_values=image_cfg_cache,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_type="serial_text_img",
                cfg_interval=[0, 1.0],
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
                timestep_shift=3.0,
                **inputs,
                cfg_text_packed_position_ids=text_cfg_inputs[
                    "cfg_packed_position_ids"
                ],
                cfg_text_packed_query_indexes=text_cfg_inputs[
                    "cfg_packed_query_indexes"
                ],
                cfg_text_key_values_lens=text_cfg_inputs[
                    "cfg_key_values_lens"
                ],
                cfg_text_packed_key_value_indexes=text_cfg_inputs[
                    "cfg_packed_key_value_indexes"
                ],
                cfg_img_packed_position_ids=image_cfg_inputs[
                    "cfg_packed_position_ids"
                ],
                cfg_img_packed_query_indexes=image_cfg_inputs[
                    "cfg_packed_query_indexes"
                ],
                cfg_img_key_values_lens=image_cfg_inputs[
                    "cfg_key_values_lens"
                ],
                cfg_img_packed_key_value_indexes=image_cfg_inputs[
                    "cfg_packed_key_value_indexes"
                ],
            )[0]
        latent = latent.reshape(
            1, height // 16, width // 16, 2, 2, 16
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(
            1, 16, height // 8, width // 8
        )
        decoded = self.vae_model.decode(latent)[0]
        if self.depth_transform is None:
            return (
                (decoded * 0.5 + 0.5)
                .clamp(0, 1)
                .permute(1, 2, 0)
                .mean(dim=-1)
            )
        return decode_depth_wrap(self.depth_transform, decoded)

    def infer_depth(
        self, images: Sequence[Image.Image], prompt: str, **kwargs
    ) -> torch.Tensor:
        original_width, original_height = images[0].size
        depth = self.inference(images, prompt, **kwargs)
        return (
            functional.interpolate(
                depth[None, None],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            .squeeze()
            .detach()
            .cpu()
        )
