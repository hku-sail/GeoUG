# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

import PIL
from PIL import Image
import torch

from data.data_utils import (
    pil_img2rgb,
    len2weight,
    create_sparse_mask,
    prepare_attention_mask_per_sample,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel.bagel import sde_step, sde_step_with_flowcps

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


def detach_context(context):
    past_key_values = context['past_key_values']
    if past_key_values is not None:
        for v in past_key_values.key_cache.values():
            v.detach_()
        for v in past_key_values.value_cache.values():
            v.detach_()
    return context


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


class BaseInterleaveInferencer:
    use_flex_attention = True
    compute_dtype = torch.bfloat16

    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        if self.model.config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def init_gen_context(self, batch_size=1):
        gen_context = {
            'kv_lens': [0] * batch_size,
            'ropes': [0] * batch_size,
            'past_key_values':
                NaiveCache(self.model.module.config.llm_config.num_hidden_layers)
                if hasattr(self.model, 'module') else
                NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @property
    def device(self):
        return self.model.device

    # @torch.no_grad()
    def update_context_text(self, texts: List[str], gen_context, return_logprobs=False, **kwargs):
        # used for interleave data, currently only support 1 data inference,
        if isinstance(texts, str):
            texts = [texts]

        prepare_prompts_fn = self.model.module.prepare_prompts if hasattr(self.model, 'module') else self.model.prepare_prompts

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = prepare_prompts_fn(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=texts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = {k: v.to(self.device) for k, v in generation_input.items()}
        output = self.model(
            past_key_values,
            **generation_input,
            use_flex_attention=self.use_flex_attention,
            return_logprobs=return_logprobs,
            forward_mode='forward_cache_update_text',
            **kwargs)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes

        if return_logprobs:
            gen_context['past_key_values'] = output[0]
            text_per_token_log_probs = output[1]
            return gen_context, text_per_token_log_probs
        else:
            gen_context['past_key_values'] = past_key_values
            return gen_context

    # @torch.no_grad()
    def update_context_image(self, images, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if not isinstance(images, list):
            images = [images]

        prepare_vae_images_fn = self.model.module.prepare_vae_images if hasattr(self.model, 'module') else self.model.prepare_vae_images
        prepare_vit_images_fn = self.model.module.prepare_vit_images if hasattr(self.model, 'module') else self.model.prepare_vit_images

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = prepare_vae_images_fn(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in generation_input.items()}
            past_key_values = self.model(
                self.vae_model, past_key_values, use_flex_attention=self.use_flex_attention, **generation_input,
                forward_mode='forward_cache_update_vae')

        if vit:
            ## update vit
            generation_input, kv_lens, ropes = prepare_vit_images_fn(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = {k: v.to(self.device) for k, v in generation_input.items()}
            past_key_values = self.model(
                past_key_values, use_flex_attention=self.use_flex_attention, **generation_input,
                forward_mode='forward_cache_update_vit')

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
            self,
            image_shapes,
            gen_context,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,

            cfg_text_precontext=None,
            cfg_img_precontext=None,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",

            num_timesteps=50,
            timestep_shift=3.0,
            **kwargs,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        if isinstance(image_shapes[0], int):
            image_shapes = [image_shapes]

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
        )
        generation_input = {k: v.to(self.device) for k, v in generation_input.items()}

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=image_shapes,
        )
        generation_input_cfg_text = {k: v.to(self.device) for k, v in generation_input_cfg_text.items()}

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=image_shapes,
        )
        generation_input_cfg_img = {k: v.to(self.device) for k, v in generation_input_cfg_img.items()}

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            use_flex_attention=self.use_flex_attention,
            **kwargs
        )

        with torch.autocast('cuda', enabled=True, dtype=torch.float32):
            images = [self.decode_image(latent, image_shape) for latent, image_shape in
                      zip(unpacked_latent, image_shapes)]
        return images

    def get_model_attr(self, var_name):
        return getattr(self.model.module, var_name) if hasattr(self.model, 'module') else getattr(self.model, var_name)

    @property
    def latent_downsample(self):
        return self.model.module.latent_downsample if hasattr(self.model, 'module') else self.model.latent_downsample

    @property
    def latent_channel(self):
        return self.model.module.latent_channel if hasattr(self.model, 'module') else self.model.latent_channel

    @property
    def max_latent_size(self):
        return self.model.module.max_latent_size if hasattr(self.model, 'module') else self.model.max_latent_size

    @property
    def latent_patch_size(self):
        return self.model.module.latent_patch_size if hasattr(self.model, 'module') else self.model.latent_patch_size

    @property
    def num_hidden_layers(self):
        if hasattr(self.model, 'module'):
            return self.model.module.config.llm_config.num_hidden_layers
        else:
            return self.model.config.llm_config.num_hidden_layers

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.latent_downsample, W // self.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size,
                                self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size,
                                w * self.model.latent_patch_size)

        image = self.vae_model.decode(latent.to(self.model.device))
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, top_p: float = None, top_k: int = None, prefix_think_token_ids=None):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids, 
            prefix_think_token_ids=prefix_think_token_ids
        )
        generation_input = move_generation_input_to_device(generation_input, self.device)
        
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            end_token_id=self.new_token_ids['eos_token_id'],
            use_flex_attention=self.use_flex_attention,
            **generation_input,
        )
        output = self.tokenizer.batch_decode(unpacked_latent.transpose(1, 0))
        output = [t.split('<|im_end|>')[0].split('<|im_start|>')[1] for t in output]

        return output

    @torch.no_grad()
    def interleave_inference(
            self,
            input_lists: List[Union[str, Image.Image]],
            think=False,
            understanding_output=False,

            max_think_token_n=1000,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=3.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.4, 1.0),
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    # cfg_text_context = self.update_context_image(input_term, cfg_text_context, vae=not understanding_output)
                        # exit(0)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature,
                                         max_length=max_think_token_n)
                output_list.append(gen_text[0])

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature,
                                             max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text[0])

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )[0]

                output_list.append(img)

        return output_list

    def __call__(
            self,
            image: Optional[Image.Image] = None,
            text: Optional[str] = None,
            **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, list):
                input_list.extend(image)
            else:
                input_list.append(image)

        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict


class InterleaveInferencer(BaseInterleaveInferencer):
    use_flex_attention = True

    @torch.no_grad()
    def gen_image(
            self,
            image_shapes,
            gen_context,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,

            cfg_text_precontext=None,
            cfg_img_precontext=None,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",

            num_timesteps=50,
            timestep_shift=3.0,
            return_log_probs: bool = False,
            generators=None,
            **kwargs,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
            generators=deepcopy(generators)
        )

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=image_shapes,
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=image_shapes,
        )

        generation_input = move_generation_input_to_device(generation_input, self.device)
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, self.device)
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, self.device)

        outputs = self.model.generate_image_with_sde(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            **kwargs,
            return_log_probs=return_log_probs,
            use_flex_attention=self.use_flex_attention,
        )

        if return_log_probs:
            unpacked_latent, all_latents, all_log_probs = outputs
            with torch.autocast('cuda', enabled=True, dtype=torch.float32):
                images = [self.decode_image(latent, image_shape) for latent, image_shape in
                          zip(unpacked_latent, image_shapes)]
            return images, all_latents, all_log_probs
        else:
            unpacked_latent = outputs
            with torch.autocast('cuda', enabled=True, dtype=torch.float32):
                images = [self.decode_image(latent, image_shape) for latent, image_shape in
                          zip(unpacked_latent, image_shapes)]
            return images

    @torch.no_grad()
    def interleave_inference(
            self,
            input_lists: List[List[Union[str, Image.Image]]],
            think=False,
            understanding_output=False,

            max_think_token_n=2048,
            do_sample=False,
            text_temperature=0.3,
            text_top_p: float = 1.0,
            text_top_k: int = 0,
            cfg_text_scale=3.0,
            cfg_img_scale=1.5,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(1024, 1024),

            noise_level: float = 0.7,
            deterministic: bool = True,
            return_log_probs: bool = False,
    ) -> List[Union[str, Image.Image]]:
        # only support one text, image output.

        # support interleaved image-text input.
        assert len(set(len(seq) for seq in input_lists)) == 1, "All sequences must have the same length. Got {}".format(
            [len(seq) for seq in input_lists])
        assert all([len(set([type(seq[i]) for seq in input_lists])) == 1 for i in
                    range(len(input_lists[0]))]), "All sequences must have the same type. Got {}".format(
            [[type(seq[i]) for seq in input_lists] for i in range(len(input_lists[0]))])
        
        seq_len = len(input_lists[0])
        batch_size = len(input_lists)

        output_list = []  # images
        output_list_latents = []
        output_list_log_probs = []
        gen_context = self.init_gen_context(batch_size)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
    
                system_prompts = [system_prompt for _ in range(batch_size)]
                gen_context = self.update_context_text(system_prompts, gen_context)
                cfg_img_context = self.update_context_text(system_prompts, cfg_img_context)

            for pos in range(seq_len):
                input_terms = [seq[pos] for seq in input_lists]

                if isinstance(input_terms[0], str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_terms, gen_context)
                    cfg_img_context = self.update_context_text(input_terms, cfg_img_context)

                elif isinstance(input_terms[0], Image.Image):
                    input_terms = [self.vae_transform.resize_transform(pil_img2rgb(img)) for img in input_terms]
                    gen_context = self.update_context_image(input_terms, gen_context, vae=not understanding_output)

                    image_shapes = [item.size[::-1] for item in input_terms]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_terms[0])}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, 
                                         temperature=text_temperature, top_p=text_top_p, top_k=text_top_k, 
                                         max_length=max_think_token_n)
                output_list.append(gen_text[0])

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, 
                                             temperature=text_temperature, top_p=text_top_p, top_k=text_top_k, 
                                             max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text[0])

                gen_output = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    noise_level=noise_level,
                    deterministic=deterministic,
                    return_log_probs=return_log_probs,
                )

                if return_log_probs:
                    img, all_latents, all_log_probs = gen_output
                    output_list.append(img[0])
                    output_list_latents.append(all_latents)
                    output_list_log_probs.append(all_log_probs)
                else:
                    img = gen_output
                    output_list.append(img[0])

        if return_log_probs:
            return output_list, output_list_latents, output_list_log_probs
        else:
            return output_list

    def __call__(
            self,
            input_lists: List[List[Union[str, Image.Image]]],
            **kargs
    ) -> List[Dict[str, Any]]:
        """Batch inference via interleave_inference.

        Args:
            input_lists: List of input sequences. Each sequence is a list of
                interleaved str/Image items. All sequences must have the same
                length and matching types at each position.
                Examples:
                    T2I:  [["a cat"], ["a dog"]]
                    Edit: [[img1, "make it sunset"], [img2, "add snow"]]
            **kargs: Forwarded to interleave_inference.

        Returns:
            List of output dicts with 'type' ('image'|'text') and 'data' keys.
        """
        output_list = self.interleave_inference(input_lists, **kargs)
        outputs = []
        for item in output_list:
            if isinstance(item, Image.Image):
                outputs.append(dict(type='image', data=item))
            elif isinstance(item, str):
                outputs.append(dict(type='text', data=item))
            else:
                raise ValueError(f"Unsupported output type: {type(item)}")
        return outputs

    def expand_kv_cache(self, past_key_values: NaiveCache, repeat_think_text: int = 0):
        past_key_values.key_cache = {k: v.repeat(repeat_think_text, 1, 1) for k, v in past_key_values.key_cache.items()}
        past_key_values.value_cache = {k: v.repeat(repeat_think_text, 1, 1) for k, v in past_key_values.value_cache.items()}
        return past_key_values

    def expand_context(self, context, repeat_think_text: int = 0):
        if repeat_think_text > 1:
            context = {
                'kv_lens': context['kv_lens'] * repeat_think_text,
                'ropes': context['ropes'] * repeat_think_text,
                'past_key_values':
                    self.expand_kv_cache(context['past_key_values'], repeat_think_text)
            }
        return context

    @torch.no_grad()
    def batch_generate_images(
            self,
            input_lists: List[List[Union[str, PIL.Image.Image]]],
            image_shape: object = (1024, 1024),
            think: bool = False,
            text_temperature: float = 0.3,
            text_top_p: float = 1.0,
            text_top_k: int = 0,
            cfg_text_scale: object = 4.0,
            cfg_img_scale: object = 1.5,
            cfg_interval: object = [0.4, 1.0],
            timestep_shift: object = 3.0,
            num_timesteps: object = 50,
            cfg_renorm_min: object = 0.0,
            cfg_renorm_type: object = "global",

            deterministic: bool = True,
            noise_level: float = 0.7,
            max_think_token_n: object = 1024,
            return_log_probs: bool = False,
            sde_window: object = None,
            generators: List[torch.Generator] = None,
            batch_forward_cfg: object = True,

            repeat_think_text: int = 0,  # repeat think text for how many times. This make a branch for the later image generation.
            prefix_think_token_ids: str = None,
            use_flowcps: bool = False,
            return_middle_context=False,
            system_prompt=None,
    ) -> Any:
        # support interleaved image-text input.
        assert len(set(len(seq) for seq in input_lists)) == 1, "All sequences must have the same length. Got {}".format(
            [len(seq) for seq in input_lists])
        assert all([len(set([type(seq[i]) for seq in input_lists])) == 1 for i in
                    range(len(input_lists[0]))]), "All sequences must have the same type. Got {}".format(
            [[type(seq[i]) for seq in input_lists] for i in range(len(input_lists[0]))])

        seq_len = len(input_lists[0])
        batch_size = len(input_lists)
        if isinstance(image_shape[0], int):
            image_shape = [image_shape for _ in range(batch_size)]

        gen_context = self.init_gen_context(batch_size)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        if think:
            system_prompt = GEN_THINK_SYSTEM_PROMPT if system_prompt is None else system_prompt
            system_prompts = [system_prompt for _ in range(batch_size)]
            gen_context = self.update_context_text(system_prompts, gen_context)
            cfg_img_context = self.update_context_text(system_prompts, cfg_img_context)

        for pos in range(seq_len):
            # Collect items at the current position
            items = [seq[pos] for seq in input_lists]

            with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
                if isinstance(items[0], str):  # All text
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(items, gen_context)
                    cfg_img_context = self.update_context_text(items, cfg_img_context)

                elif isinstance(items[0], PIL.Image.Image):  # All images
                    input_terms = [self.vae_transform.resize_transform(pil_img2rgb(img)) for img in items]
                    gen_context = self.update_context_image(input_terms, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                    image_shape = [item.size[::-1] for item in input_terms]
                else:
                    raise ValueError(f"Invalid input type: {type(items[0])}")

        if return_middle_context:
            middle_gen_context = deepcopy(gen_context)
            middle_cfg_text_context = deepcopy(cfg_text_context)
            middle_cfg_img_context = deepcopy(cfg_img_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
            text_per_token_log_probs = None
            think_texts = None

            if think:
                think_texts = self.gen_text(gen_context,
                                            max_length=max_think_token_n,
                                            do_sample=True,
                                            temperature=text_temperature,
                                            top_p=text_top_p,
                                            top_k=text_top_k,
                                            prefix_think_token_ids=prefix_think_token_ids)

                gen_context = self.update_context_text(
                    think_texts, gen_context,
                    temperature=text_temperature, return_logprobs=return_log_probs)
                if return_log_probs:
                    gen_context, text_per_token_log_probs = gen_context

                if repeat_think_text > 1:
                    input_lists = [deepcopy(item) for _ in range(repeat_think_text) for item in input_lists]
                    gen_context = self.expand_context(gen_context, repeat_think_text)
                    cfg_text_context = self.expand_context(cfg_text_context, repeat_think_text)
                    cfg_img_context = self.expand_context(cfg_img_context, repeat_think_text)
                    image_shape = image_shape * repeat_think_text
                    think_texts = think_texts * repeat_think_text

                    if isinstance(text_per_token_log_probs, tuple):
                        text_per_token_log_probs = text_per_token_log_probs[0]
                    text_per_token_log_probs = torch.cat([text_per_token_log_probs] * repeat_think_text, dim=0)

            if return_middle_context and repeat_think_text > 1:
                middle_gen_context = self.expand_context(middle_gen_context, repeat_think_text)
                middle_cfg_text_context = self.expand_context(middle_cfg_text_context, repeat_think_text)
                middle_cfg_img_context = self.expand_context(middle_cfg_img_context, repeat_think_text)

            gen_output = self.gen_image(
                image_shape,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,

                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,

                batch_forward_cfg=batch_forward_cfg,
                noise_level=noise_level,
                deterministic=deterministic,
                sde_window=sde_window,
                return_log_probs=return_log_probs,
                generators=generators,
                use_flowcps=use_flowcps,
            )

            if return_log_probs:
                images, all_latents, image_log_probs = gen_output
            else:
                images = gen_output

        if not return_log_probs:
            return images, think_texts

        if return_middle_context:
            return images, all_latents, image_log_probs, think_texts, text_per_token_log_probs, \
                (middle_gen_context, middle_cfg_text_context, middle_cfg_img_context)

        return images, all_latents, image_log_probs, think_texts, text_per_token_log_probs

    @torch.no_grad()
    def batch_generate_texts(
            self,
            input_lists: List[List[Union[str, PIL.Image.Image]]],
            think: bool = False,
            text_temperature: float = 0.3,
            text_top_p: float = 1.0,
            text_top_k: int = 0,
            max_think_token_n: object = 2048,
            prefix_think_token_ids: str = None,
            system_prompt = None,
            return_log_probs: bool = False,
            return_middle_context=False,
    ) -> Any:
        # support interleaved image-text input.
        assert len(set(len(seq) for seq in input_lists)) == 1, "All sequences must have the same length. Got {}".format(
            [len(seq) for seq in input_lists])
        assert all([len(set([type(seq[i]) for seq in input_lists])) == 1 for i in
                    range(len(input_lists[0]))]), "All sequences must have the same type. Got {}".format(
            [[type(seq[i]) for seq in input_lists] for i in range(len(input_lists[0]))])

        seq_len = len(input_lists[0])
        batch_size = len(input_lists)

        gen_context = self.init_gen_context(batch_size)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        if think:
            system_prompt = GEN_THINK_SYSTEM_PROMPT if system_prompt is None else system_prompt
            system_prompts = [system_prompt for _ in range(batch_size)]
            gen_context = self.update_context_text(system_prompts, gen_context)
            cfg_img_context = self.update_context_text(system_prompts, cfg_img_context)

        for pos in range(seq_len):
            # Collect items at the current position
            items = [seq[pos] for seq in input_lists]

            with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
                if isinstance(items[0], str):  # All text
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(items, gen_context)
                    cfg_img_context = self.update_context_text(items, cfg_img_context)

                elif isinstance(items[0], PIL.Image.Image):  # All images
                    input_terms = [self.vae_transform.resize_transform(pil_img2rgb(img)) for img in items]
                    gen_context = self.update_context_image(input_terms, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Invalid input type: {type(items[0])}")

        if return_middle_context:
            middle_gen_context = deepcopy(gen_context)
            middle_cfg_text_context = deepcopy(cfg_text_context)
            middle_cfg_img_context = deepcopy(cfg_img_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=self.compute_dtype):
            think_texts = None
            think_texts_per_token_log_probs = None

            if think:
                think_texts = self.gen_text(gen_context,
                                            max_length=max_think_token_n,
                                            do_sample=True,
                                            temperature=text_temperature,
                                            top_p=text_top_p,
                                            top_k=text_top_k,
                                            prefix_think_token_ids=prefix_think_token_ids)

                gen_context = self.update_context_text(
                    think_texts, gen_context,
                    temperature=text_temperature, return_logprobs=return_log_probs)
                if return_log_probs:
                    gen_context, think_texts_per_token_log_probs = gen_context

            output_texts = self.gen_text(gen_context,
                                        max_length=max_think_token_n,
                                        do_sample=True,
                                        temperature=text_temperature,
                                        top_p=text_top_p,
                                        top_k=text_top_k,
                                        prefix_think_token_ids=prefix_think_token_ids)

            gen_context = self.update_context_text(
                output_texts, gen_context,
                temperature=text_temperature, return_logprobs=return_log_probs)
            if return_log_probs:
                gen_context, output_texts_per_token_log_probs = gen_context

        if not return_log_probs:
            return output_texts, think_texts

        if return_middle_context:
            return output_texts, output_texts_per_token_log_probs, think_texts, think_texts_per_token_log_probs, (middle_gen_context, middle_cfg_text_context, middle_cfg_img_context)

        return output_texts, output_texts_per_token_log_probs, think_texts, think_texts_per_token_log_probs

    def compute_text_logprob(
            self, 
            think_text, 
            gen_context,
            temperature=1.0,
            return_context=True,
            compute_entropy=False,
            model=None):
        """
        Compute the log probability of a transition for the Bagel model.
        """
        model = model if model is not None else self.model

        if isinstance(think_text, str):
            think_text = [think_text]

        prepare_prompts_fn = self.model.module.prepare_prompts if hasattr(self.model, 'module') else self.model.prepare_prompts

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = prepare_prompts_fn(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=think_text,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = {k: v.to(self.device) for k, v in generation_input.items()}
        output = model(
            past_key_values,
            **generation_input,
            use_flex_attention=self.use_flex_attention,
            temperature=temperature,
            compute_entropy=compute_entropy,
            return_logprobs=True,
            forward_mode='forward_cache_update_text')
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = output[0]
        text_per_token_log_probs = output[1]
        text_token_lens = output[2]
        text_entropy = output[3]

        # gen_context = self.update_context_text(
        #     think_text, gen_context, return_logprobs=True,
        # )
        # text_per_token_log_probs = gen_context.pop('text_per_token_log_probs')
        
        if return_context:
            gen_context = detach_context(gen_context)
            return text_per_token_log_probs, text_token_lens, text_entropy, gen_context

        return text_per_token_log_probs, text_token_lens

    def get_timesteps(self, num_timesteps, timestep_shift, device):
        if hasattr(self.model, 'get_timesteps'):
            timesteps = self.model.get_timesteps(num_timesteps, timestep_shift=timestep_shift, device=device)
        else:
            timesteps = self.model.module.get_timesteps(
                num_timesteps, timestep_shift=timestep_shift, device=device)
        return timesteps

    def compute_image_logprob(self, latents, next_latents, step_idx,
                                gen_context, cfg_text_context, cfg_img_context,
                                image_shape,
                                num_timesteps, timestep_shift,
                                cfg_text_scale, cfg_img_scale, noise_level, cfg_interval,
                                cfg_renorm_min, cfg_renorm_type, cfg_type, 
                                use_flowcps=False,
                                model=None):
        """
        Compute the log probability of a transition for the Bagel model.

        yield the middle grad out.
        """
        model = model if model is not None else self.model
        timesteps = self.get_timesteps(
                num_timesteps, timestep_shift=timestep_shift, device=latents.device)

        sigma = timesteps[step_idx]
        sigma_prev = timesteps[step_idx + 1]
        dt = sigma_prev - sigma
        x_t = latents

        if sigma > cfg_interval[0] and sigma <= cfg_interval[1]:
            cfg_text_scale_ = cfg_text_scale
            cfg_img_scale_ = cfg_img_scale
        else:
            cfg_text_scale_ = 1.0
            cfg_img_scale_ = 1.0

        # Get batch size and device
        device = latents.device

        batch_size = len(gen_context['kv_lens'])
        model = self.model.module if hasattr(self.model, 'module') else self.model
        image_shape = [image_shape] * batch_size

        # text cfg
        cfg_text_past_key_values = cfg_text_context['past_key_values']
        generation_input_cfg_text = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context['kv_lens'], curr_rope=cfg_text_context['ropes'],
            image_sizes=image_shape,
        )
        generation_input_cfg_text = {k: v.to(device) for k, v in generation_input_cfg_text.items()}

        # img cfg
        cfg_img_past_key_values = cfg_img_context['past_key_values']
        generation_input_cfg_img = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context['kv_lens'], curr_rope=cfg_img_context['ropes'],
            image_sizes=image_shape,
        )
        generation_input_cfg_img = {k: v.to(device) for k, v in generation_input_cfg_img.items()}

        # Prepare for image generation
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        new_rope = gen_context['ropes']
        generation_input = model.prepare_vae_latent(
            curr_kvlens=kv_lens, curr_rope=new_rope,
            image_sizes=image_shape,
            new_token_ids=self.new_token_ids,
        )
        generation_input.pop('packed_init_noises')
        generation_input = {k: v.to(device) for k, v in generation_input.items()}

        # Use the model's forward method to get predictions
        # but for a single step calculation
        timestep = torch.tensor([sigma] * latents.shape[0],
                                device=latents.device)
        
        model_output = model(
            x_t=latents.to(self.compute_dtype),
            timestep=timestep,
            **generation_input,

            past_key_values=past_key_values,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            # cfg_text
            cfg_text_scale=cfg_text_scale_,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            # cfg_img
            cfg_img_scale=cfg_img_scale_,
            cfg_img_past_key_values=cfg_img_past_key_values,
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            cfg_type=cfg_type,
            use_flex_attention=self.use_flex_attention,
            forward_mode='batch_forward_flow',
        )

        # Get v_t from model output
        v_t = model_output.to(latents.device)

        # Calculate log probability based on the SDE step
        sigma_max = timesteps[1]

        if use_flowcps:
            x_t, image_log_prob, x_t_mean, std_dev_t = sde_step_with_flowcps(
                x_t, v_t, sigma, sigma_prev, sigma_max, dt, noise_level, prev_sample=next_latents)
        else:
            x_t, image_log_prob, x_t_mean, std_dev_t = sde_step(
                x_t, v_t, sigma, sigma_prev, sigma_max, dt, noise_level, prev_sample=next_latents)

        # Handle packed format: group log_probs by sample using packed_seqlens
        if 'packed_seqlens' in generation_input:
            # For VAE tokens, we need to exclude the start/end tokens from each sample
            # Each sample has (packed_seqlens[i] - 2) VAE tokens
            vae_seqlens = generation_input['packed_seqlens'] - 2

            # Split log_prob according to VAE sequence lengths and sum over spatial dimensions
            image_log_prob = torch.stack([image_log_prob.mean() for image_log_prob in image_log_prob.split(vae_seqlens.cpu().tolist(), dim=0)])
        else:
            image_log_prob = image_log_prob.view(batch_size, -1).mean(dim=1)

        return image_log_prob, x_t_mean, std_dev_t, model_output

    def compute_image_logprob_fm(self, clean_latents, step_idx,
                                gen_context,
                                image_shape,
                                num_timesteps, timestep_shift,
                                model=None, noise=None):
        """
        Compute the log probability of a transition for the Bagel model.

        yield the middle grad out.
        """
        model = model if model is not None else self.model
        unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
        timesteps =self.get_timesteps(
                num_timesteps, timestep_shift=timestep_shift, device=clean_latents.device)

        sigma = timesteps[step_idx]
        if noise is None:
            noise = torch.randn_like(clean_latents)
        noisy_latent = (1 - sigma) * clean_latents + sigma * noise

        # Get batch size and device
        device = clean_latents.device

        batch_size = len(gen_context['kv_lens'])
        image_shape = [image_shape] * batch_size

        # Prepare for image generation
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        new_rope = gen_context['ropes']
        generation_input = unwrapped_model.prepare_vae_latent(
            curr_kvlens=kv_lens, curr_rope=new_rope,
            image_sizes=image_shape,
            new_token_ids=self.new_token_ids,
        )
        generation_input.pop('packed_init_noises')
        generation_input = {k: v.to(device) for k, v in generation_input.items()}

        # Use the model's forward method to get predictions
        # but for a single step calculation
        timestep = torch.tensor([sigma] * clean_latents.shape[0], device=device)
        
        model_output = model(
            x_t=noisy_latent.to(self.compute_dtype),
            timestep=timestep,
            **generation_input,

            past_key_values=past_key_values,
            use_flex_attention=self.use_flex_attention,
            forward_mode='forward_flow',
        )

        # Get v_t from model output
        v_t = model_output.to(device)

        target = noise - clean_latents
        image_log_prob = - (v_t - target) ** 2

        vae_seqlens = generation_input['packed_seqlens'] - 2

        std_dev_t = torch.sqrt(sigma / (1 - torch.clamp(sigma, 0, 0.99)))*0.7
        prev_sample_mean = noisy_latent - sigma * v_t

        # Split log_prob according to VAE sequence lengths and sum over spatial dimensions
        image_log_prob = torch.stack([image_log_prob.mean() for image_log_prob in image_log_prob.split(vae_seqlens.cpu().tolist(), dim=0)])

        return image_log_prob, prev_sample_mean, std_dev_t, model_output