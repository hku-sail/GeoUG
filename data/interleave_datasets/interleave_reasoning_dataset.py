import copy
import io
import json
import os
import re
import base64
import torch
import pyarrow.parquet as pq
import random
import numpy as np
import torchvision
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..constant import (
    IMAGE_FLAG,
    INTERLEAVE_COT_PROMPT,
    DEPTH_COT_SYSTEM_PROMPT,
    DEPTH_FLAG,
    ORACLE_TAG_PAIRS,
)
from ..transforms_depth import DepthTransform, DepthTransformBase

from torchvision import transforms

from loguru import logger

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# capture-group split so the flag tokens themselves are kept in the result
_FLAG_SPLIT_PATTERN = re.compile(
    f"({re.escape(IMAGE_FLAG)}|{re.escape(DEPTH_FLAG)})"
)

# ORACLE_TAG_PAIRS is defined in data.constant (single source of truth);
# the regex here just compiles them into a span-detector. Both the tags
# themselves and their payloads end up loss-masked (see _iter_text_segments).
_ORACLE_SEGMENT_PATTERN = re.compile(
    '|'.join(
        f"{re.escape(open_tag)}.*?{re.escape(close_tag)}"
        for open_tag, close_tag in ORACLE_TAG_PAIRS
    ),
    flags=re.DOTALL,
)


def _iter_text_segments(text, default_has_loss):
    """Yield (segment, has_loss) chunks from `text`.

    Any span matching an entry in ORACLE_TAG_PAIRS (inclusive of the tags
    themselves) is yielded with has_loss=0. Everything else inherits
    `default_has_loss`. Empty / whitespace-only chunks are skipped.
    """
    if default_has_loss == 0:
        # whole turn is already no-loss (e.g. user/human), nothing to mask
        if text and text.strip():
            yield text, 0
        return

    cursor = 0
    for m in _ORACLE_SEGMENT_PATTERN.finditer(text):
        if m.start() > cursor:
            pre = text[cursor:m.start()]
            if pre.strip():
                yield pre, default_has_loss
        span = m.group(0)
        if span.strip():
            yield span, 0
        cursor = m.end()
    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            yield tail, default_has_loss


class InterleaveReasoningDatasetDepth(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, 
        vit_transform, vit_gen_transform,
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        depth_transform=None, vit_depth_transform=None,
        dataset_given_name="", **kwargs
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(
            dataset_name=dataset_name, transform=transform, tokenizer=tokenizer, 
            vit_transform=vit_transform, data_dir_list=data_dir_list, 
            num_used_data=num_used_data, parquet_info=parquet_info, 
            local_rank=local_rank, world_size=world_size, 
            num_workers=num_workers, data_status=data_status
            )

        # default image stride is 16 (flux vae encode)
        self.depth_transform = depth_transform
        assert self.depth_transform is not None
        # self.depth_normalize = ScaleShiftDepthNormalizer(norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02)
        if isinstance(self.depth_transform, DepthTransformBase):
            self.depth_normalize = self.depth_transform.normalize
        else:
            raise ValueError("depth_transform must be a DepthTransformBase")
        
        self.depth_min = 1e-5
        self.dataset_given_name = dataset_given_name
        self.include_vit_feature = kwargs.get('include_vit_feature', True)
        logger.info("#"*10 + f"include_vit_feature: {self.include_vit_feature} in {dataset_given_name}")
        # 在每条样本最前面注入 Depth-CoT system prompt（no-loss）。
        # 评测端务必用同一份 DEPTH_COT_SYSTEM_PROMPT，保持 train/eval 一致。
        # 默认关闭，需要时在 dataset 配置里传 use_system_prompt=True 打开。
        self.use_system_prompt = kwargs.get('use_system_prompt', False)
        logger.info("#"*10 + f"use_system_prompt: {self.use_system_prompt} in {dataset_given_name}")
        # this is used for generation
        self.vit_gen_transform = vit_gen_transform
        # 把生成出来的深度图**同时**送进理解 ViT，让后续裁决文本能读到深度图本身，
        # 而不只是工具返回的标量读数。默认关闭；打开需要在 dataset 配置里同时给
        # vit_depth_transform_args（与 depth_transform_args 同一套 normalization，
        # 只把 image_stride 换成 ViT 的 14）。
        self.vit_depth_transform = vit_depth_transform
        self.depth_need_vit = kwargs.get('depth_need_vit', False)
        if self.depth_need_vit:
            assert self.vit_depth_transform is not None, (
                "depth_need_vit=True but vit_depth_transform_args is missing "
                f"from the dataset config of {dataset_given_name}"
            )
        logger.info("#"*10 + f"depth_need_vit: {self.depth_need_vit} in {dataset_given_name}")

    def change_format(self, data, image_input_list):
        """
        Build the element list for one interleave-reasoning sample.

        Layout rules:
          * <|image_pad|>  (input RGB, lives in user turn)
              -> understanding context only: ViT_und features (high-res,
                 dynamic). No VAE here, enable_cfg=False so CFG never drops
                 the grounding context.
          * <|depth_pad|>  (depth target, lives in assistant turn)
              -> right BEFORE the depth target we re-inject the most-recent
                 source RGB as **generation condition**: VAE + ViT_gen
                 (lower-res, fixed), both enable_cfg=True so CFG-img can
                 drop them at inference. Then we emit the depth target
                 itself (loss=1) plus a no-loss VAE clone so the follow-up
                 reasoning text can attend to the generated depth.
          * Any text span wrapped in ORACLE_TAG_PAIRS (e.g.
            ``<tool_result>...</tool_result>``) is split out as a separate
            text element with has_loss=0, so the model never tries to
            memorise tool / environment returns even when those spans live
            inside an assistant turn.
        """
        elements = []
        image_idx = 0
        last_rgb = None  # most-recent source RGB, reused as gen condition

        def _emit_text(raw_text, default_has_loss):
            """Split raw_text by oracle tags and push one text element per
            non-empty segment with the appropriate has_loss."""
            for seg, seg_loss in _iter_text_segments(raw_text, default_has_loss):
                seg = seg.strip()
                if not seg:
                    continue
                elements.append({
                    'type': 'text',
                    'has_loss': seg_loss,
                    'text': seg,
                })

        for conversation in data['conversations']:
            has_loss = 0 if conversation['from'] == 'human' else 1
            conv_value = conversation['value']

            if IMAGE_FLAG not in conv_value and DEPTH_FLAG not in conv_value:
                _emit_text(conv_value, has_loss)
                continue

            for part in _FLAG_SPLIT_PATTERN.split(conv_value):
                if part == IMAGE_FLAG:
                    assert image_idx < len(image_input_list), \
                        f"<|image_pad|> overflows image_input_list at idx {image_idx}"
                    item = image_input_list[image_idx]
                    assert item['type'] == 'rgb', \
                        f"<|image_pad|> expects rgb, got {item['type']}"
                    last_rgb = item['image']
                    elements.append({
                        'type': 'image',
                        'has_loss': 0,
                        'image': item['image'],
                        'need_vae': False,
                        'need_vit': True,
                        'which_vit': 'und',
                        'enable_cfg': False,
                    })
                    image_idx += 1

                elif part == DEPTH_FLAG:
                    assert image_idx < len(image_input_list), \
                        f"<|depth_pad|> overflows image_input_list at idx {image_idx}"
                    item = image_input_list[image_idx]
                    assert item['type'] == 'depth', \
                        f"<|depth_pad|> expects depth, got {item['type']}"
                    assert last_rgb is not None, \
                        "saw <|depth_pad|> with no preceding RGB; cannot supply gen condition"

                    # 1) gen condition: VAE + ViT_gen on the source RGB
                    elements.append({
                        'type': 'image',
                        'has_loss': 0,
                        'image': last_rgb,
                        'need_vae': True,
                        'need_vit': True,
                        'which_vit': 'gen',
                        'enable_cfg': True,
                    })
                    # 2) depth target (loss=1) + clone (loss=0) so the
                    #    follow-up tool/answer text can read the depth
                    # 这里需要确定是不是需要 vae input
                    elements.append({
                        'type': 'depth',
                        'has_loss': has_loss,
                        'depth': item['image'],
                        'valid_mask': item.get('valid_mask', None),
                        'need_vae': True,
                        'need_vit': self.depth_need_vit,
                        'vit_image': item.get('vit_image', None),
                    })
                    image_idx += 1

                elif part:
                    _emit_text(part, has_loss)

        return elements

    def parse_row(self, row):

        image_info_list = row['images']

        # Load every image; for depth we additionally run resize + normalize
        # + 3-channel replication here so _add_depth can treat it as a ready
        # VAE-ready tensor and so we can derive a valid_mask for loss masking.
        image_input_list = []
        for image_info in image_info_list:
            image_type = image_info['type']
            if image_type == 'rgb':
                rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_info['bytes'])))
                image_input_list.append({'image': rgb_image, 'type': image_type})
            elif image_type == 'depth':
                depth_raw_pil = Image.open(io.BytesIO(image_info['bytes']))  # meters
                depth_metric = torch.from_numpy(
                    np.asarray(depth_raw_pil).copy()
                ).float().unsqueeze(0)  # (1, H, W), still in meters
                depth_raw = self.depth_transform.resize_transform(depth_metric)
                # valid_mask = depth_raw > self.depth_min
                valid_mask = None
                depth_raw_norm = self.depth_normalize(depth_raw)
                if depth_raw_norm.shape[0] == 1:
                    depth_norm_stack = depth_raw_norm.repeat(3, 1, 1)
                elif depth_raw_norm.shape[0] == 3:
                    depth_norm_stack = depth_raw_norm
                else:
                    raise ValueError(
                        f"unexpected depth_raw_norm shape {depth_raw_norm.shape}"
                    )
                depth_entry = {
                    'image': depth_norm_stack,
                    'valid_mask': valid_mask,
                    'type': image_type,
                }
                if self.depth_need_vit:
                    # 同一份 metric 深度，换成 ViT 的 patch 网格（stride 14）再走
                    # 同一个 disparity 编码。y=(d-c)/(d+c) 无参数、不依赖 per-image
                    # 统计，值域 [-1, 1]，与 ImageTransform 的 mean/std=0.5 输出一致，
                    # 所以 SigLIP 侧不需要额外归一化，train / eval 的编码也天然一致。
                    depth_entry['vit_image'] = self.vit_depth_transform(depth_metric)
                image_input_list.append(depth_entry)

        elements = self.change_format(row, image_input_list)

        # prepare data
        data = self._init_data()
        if self.use_system_prompt:
            data = self._add_text(
                data, DEPTH_COT_SYSTEM_PROMPT, need_loss=False, enable_cfg=False
            )
        for item in elements:
            if item['type'] == 'text':
                data = self._add_text(
                    data, item['text'],
                    need_loss=item['has_loss'],
                    enable_cfg=False,
                )
            elif item['type'] == 'image':
                data = self._add_image(
                    data, item['image'],
                    need_loss=False,
                    need_vae=item['need_vae'],
                    need_vit=item['need_vit'],
                    which_vit=item.get('which_vit', 'und'),
                    enable_cfg=item.get('enable_cfg', True),
                )
            elif item['type'] == 'depth':
                data = self._add_depth(
                    data, item['depth'],
                    valid_mask=item.get('valid_mask', None),
                    need_loss=item['has_loss'],
                    need_vae=item['need_vae'],
                    need_vit=item['need_vit'],
                    vit_image=item.get('vit_image', None),
                )

        has_loss = [item['loss'] for item in data['sequence_plan']]
        if sum(has_loss) == 0:
            print(f'No loss defined, skipped.')
            return dict()

        return data

    def _add_depth(self, data, image, valid_mask=None, need_loss=False,
                   need_vae=False, need_vit=False, vit_image=None,
                   enable_cfg=True):

        image_tensor = image  # shape=(3, H, W), already resized/normalised

        if need_loss:
            sequence_plan = {
                'type': 'vae_image',
                'enable_cfg': 0,
                'loss': 1,
                'special_token_loss': 0,
                'special_token_label': None,
            }
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

            if valid_mask is not None:
                # downsample the mask to the VAE patch grid and flatten so
                # the trainer can mask invalid pixels out of the depth loss
                sequence_plan['need_mask'] = 1
                invalid_mask = ~valid_mask
                valid_mask_downsample = ~torch.max_pool2d(
                    invalid_mask.float(),
                    self.transform.stride,
                    self.transform.stride,
                ).bool()
                valid_mask_patchified = valid_mask_downsample.float().squeeze(0).reshape(-1)
                assert valid_mask_patchified.sum() > 100, \
                    "invalid depth sample (too few valid pixels)"
                data['mask_tensor_list'].append(valid_mask_patchified)

            data['sequence_plan'].append(sequence_plan)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            # 放在 VAE clone 之后，即「深度图已经生成出来」的位置，这样后面的
            # 工具读数 / 裁决文本能 attend 到深度图本身。enable_cfg 与 clone 一致。
            assert vit_image is not None, \
                "need_vit=True for depth but no ViT-grid tensor was supplied"
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            height, width = vit_image.shape[1:]
            data['num_tokens'] += width * height // self.vit_depth_transform.stride ** 2
            data['image_tensor_list'].append(vit_image)

        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit,
                   enable_cfg=True, which_vit='und'):
        """
        which_vit:
          - 'und': self.vit_transform     (high-res dynamic, used for VLM
                                           understanding / grounding)
          - 'gen': self.vit_gen_transform (gen-side ViT, paired with VAE
                                           latent grid for the generation
                                           expert at the depth boundary)
        """
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            if which_vit == 'gen':
                assert self.vit_gen_transform is not None, \
                    "which_vit='gen' but vit_gen_transform is None"
                vit_t = self.vit_gen_transform
            else:
                vit_t = self.vit_transform

            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = vit_t(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // vit_t.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
            'mask_tensor_list': [],
        }
        return data



if __name__ == "__main__":
    from ..transforms import ImageTransform
    from transformers import Qwen2Tokenizer
    from ..transforms_depth import DepthTransform, DepthTransformBase, build_depth_transform
    import yaml
    from pathlib import Path

    transform_config = Path(__file__).resolve().parents[2] / "configs/rl/depth_cot_multi_dsp.yaml"
    transform_config = yaml.safe_load(open(transform_config))
    depth_transform = build_depth_transform(transform_config['depth_transform_args'])
    vae_transform = ImageTransform(**transform_config['image_transform_args'])
    # 这里需要给两个不同的 vit transform
    vit_transform = ImageTransform(**transform_config['vit_image_transform_args'])
    vit_gen_transform = ImageTransform(**transform_config['vit_image_gen_transform_args'])

    parquet_info_json = os.environ["PARQUET_INFO_PATH"]
    parquet_info = json.load(open(parquet_info_json))
    dataset_name = "depth_cot_ade20k"
    tokenizer = Qwen2Tokenizer.from_pretrained(os.environ["MODEL_PATH"])

    data_dir_list = [os.environ["DATASET_PATH"]]
    dataset = InterleaveReasoningDatasetDepth(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        transform=vae_transform,
        depth_transform=depth_transform,
        vit_transform=vit_transform,
        vit_gen_transform=vit_gen_transform,
        data_dir_list=data_dir_list,
        parquet_info=parquet_info,
        num_used_data=[50],
        world_size=1,
        num_workers=0,
    )

    # ---------- debug helpers ----------
    NUM_DEBUG_SAMPLES = 2
    DEBUG_OUT_ROOT = Path("/tmp/interleave_reasoning_debug")
    DEBUG_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    def _save_tensor_as_png(t: torch.Tensor, path: Path):
        """Per-tensor min-max normalize -> (H, W, 3) uint8 PNG.

        Works for both RGB tensors (already in roughly [-1, 1]) and the depth
        3-channel stack (which can hold multiscale disparities with arbitrary
        range). Min-max keeps things visible no matter the normalization.
        """
        t = t.detach().cpu().float()
        if t.dim() == 3:
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            elif t.shape[0] >= 3:
                t = t[:3]
        tmin, tmax = t.min(), t.max()
        if tmax > tmin:
            t = (t - tmin) / (tmax - tmin)
        else:
            t = torch.zeros_like(t)
        arr = (t * 255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(path)

    def dump_sample(sample: dict, tokenizer, out_dir: Path):
        """Pretty-print a sample's sequence_plan + dump all tensors to disk."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seq = sample['sequence_plan']
        text_ids_list = sample['text_ids_list']
        image_tensor_list = sample['image_tensor_list']
        mask_tensor_list = sample.get('mask_tensor_list', [])

        ti = ii = mi = 0
        log_lines = []

        header = (
            f"{'i':>3} | {'type':<10} | {'loss':>4} | {'cfg':>3} | "
            f"{'shape':<18} | extra"
        )
        sep = '-' * max(len(header), 90)
        for line in (header, sep):
            print(line)
            log_lines.append(line)

        for idx, sp in enumerate(seq):
            t = sp['type']
            loss = sp.get('loss', 0)
            cfg = sp.get('enable_cfg', 0)
            need_mask = sp.get('need_mask', 0)
            shape_str, preview, full_text = '', '', None

            if t == 'text':
                ids = text_ids_list[ti]
                full_text = tokenizer.decode(ids)
                shape_str = f"len={len(ids)}"
                preview = full_text.replace('\n', '\\n')
                if len(preview) > 100:
                    preview = preview[:100] + '...'
                ti += 1
            elif t in ('vae_image', 'vit_image'):
                img = image_tensor_list[ii]
                shape_str = 'x'.join(str(s) for s in img.shape)
                fname = f"{idx:02d}_{t}_loss{loss}_cfg{cfg}.png"
                _save_tensor_as_png(img, out_dir / fname)
                preview = f"-> {fname}  (min={img.min().item():.3f}, max={img.max().item():.3f})"
                ii += 1
            else:
                preview = f"<unknown type {t}>"

            if need_mask:
                m = mask_tensor_list[mi]
                valid = int(m.sum().item())
                total = int(m.numel())
                preview = f"{preview} | mask {valid}/{total} ({valid/total:.1%})"
                mi += 1

            row = (
                f"{idx:>3} | {t:<10} | {loss:>4} | {cfg:>3} | "
                f"{shape_str:<18} | {preview}"
            )
            print(row)
            log_lines.append(row)
            if full_text is not None:
                log_lines.append(f"    text>>> {full_text!r}")

        for line in (sep,):
            print(line)
            log_lines.append(line)

        summary = (
            f"num_tokens={sample.get('num_tokens', '?')}  "
            f"text_entries={ti}  image_entries={ii}  mask_entries={mi}  "
            f"image_tensor_list_len={len(image_tensor_list)}  "
            f"mask_tensor_list_len={len(mask_tensor_list)}"
        )
        print(summary)
        log_lines.append(summary)

        # sanity: counts on both sides must match
        assert ti == len(text_ids_list), \
            f"text count mismatch: seq={ti} vs list={len(text_ids_list)}"
        assert ii == len(image_tensor_list), \
            f"image count mismatch: seq={ii} vs list={len(image_tensor_list)}"
        assert mi == len(mask_tensor_list), \
            f"mask count mismatch: seq={mi} vs list={len(mask_tensor_list)}"

        (out_dir / 'sample.log').write_text('\n'.join(log_lines) + '\n')
        print(f"-> artifacts: {out_dir.resolve()}")
    # ---------- /debug helpers ----------

    last_sample = None
    for sample_idx, sample in enumerate(dataset):
        sample_dir = DEBUG_OUT_ROOT / f"sample_{sample_idx:02d}"
        print(f"\n===== sample {sample_idx} ({sample.get('data_indexes', {}).get('file_path', '?')}) =====")
        dump_sample(sample, tokenizer, sample_dir)
        last_sample = sample
        if sample_idx + 1 >= NUM_DEBUG_SAMPLES:
            break

    print(f"\nAll debug artifacts written under: {DEBUG_OUT_ROOT.resolve()}")
    print("Dropping into pdb for interactive inspection of `last_sample` ...")
    import pdb; pdb.set_trace()
