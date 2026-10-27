import copy
import io
import json
import os
import base64
import torch
import pyarrow.parquet as pq
import random
import numpy as np
import torchvision
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..constant import IMAGE_FLAG, INTERLEAVE_COT_PROMPT, DEPTH_PROMPT_TEMPLATE

from torchvision import transforms

from loguru import logger

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class DepthIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        depth_transform=None, dataset_given_name="", **kwargs
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
        self.depth_normalize = self.depth_transform.normalize
        self.depth_min = 1e-5
        self.dataset_given_name = dataset_given_name
        self.include_vit_feature = kwargs.get('include_vit_feature', True)
        logger.info("#"*10 + f"include_vit_feature: {self.include_vit_feature} in {dataset_given_name}")

    def parse_row(self, row):
        
        image_data = row['images'][0]
        depth_data = row['depth'][0]
        mask_data = row['mask'][0]
        rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_data['bytes'])))
        # read depth data
        depth_raw_pil = Image.open(io.BytesIO(depth_data['bytes']))
        mask_raw_pil = Image.open(io.BytesIO(mask_data['bytes']))

        # aug
        if random.random() < 0.5:
            rgb_image = torchvision.transforms.functional.hflip(rgb_image)
            depth_raw_pil = torchvision.transforms.functional.hflip(depth_raw_pil)
            mask_raw_pil = torchvision.transforms.functional.hflip(mask_raw_pil)

        depth_raw = torch.from_numpy(np.asarray(depth_raw_pil).copy()).float().unsqueeze(0) # shape=(1, h, w)
        valid_mask = torch.from_numpy(np.asarray(mask_raw_pil).copy()).unsqueeze(0).bool()  # NOTE: uint16 will be 0 during resize

        # set the depth to zero according to mask
        depth_raw[~valid_mask] = 0.

        # TODO: we need to do paried data augmentations, such as flip here
        depth_raw = self.depth_transform.resize_transform(depth_raw)  # resize
        valid_mask = depth_raw > self.depth_min
        depth_raw_norm = self.depth_normalize(depth_raw, valid_mask=valid_mask)  # normalize
        if depth_raw_norm.shape[0] == 1:
            # stack depth to shape=(3, h, w)
            depth_norm_stack = depth_raw_norm.repeat(3, 1, 1)
        elif depth_raw_norm.shape[0] == 3:
            depth_norm_stack = depth_raw_norm
        else:
            raise ValueError(f"the depth raw norm is wrong")
        prompt = random.choice(DEPTH_PROMPT_TEMPLATE)
        data = self._init_data()
        # assert len(depth_norm_stack) == 1, "we only support monocular depth estimation now"
        # change the gt to channel of 4, one for mask
        # here, we need to set the vae or vit features
        data = self._add_image(data, rgb_image, need_loss=False, need_vae=True, need_vit=self.include_vit_feature)  # condition
        data = self._add_text(data, prompt, need_loss=False)
        data = self._add_depth(data, depth_norm_stack, valid_mask=valid_mask, need_loss=True, need_vae=False, need_vit=False) # target
        
        return data

    def _add_depth(self, data, image, valid_mask, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        image_tensor = image  # shape=(B, H, W)

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image',
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'need_mask': 1
                }
            )

            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)
            # downsample and patchify here
            invalid_mask = ~valid_mask
            valid_mask_downsample = ~torch.max_pool2d(invalid_mask.float(), self.transform.stride, self.transform.stride).bool()
            valid_mask_downsample = valid_mask_downsample.float()
            valid_mask_patchified = valid_mask_downsample.squeeze(0).reshape(-1)  # (1, h, w) -> (hw, 1)
            assert valid_mask_patchified.sum() > 100, f"invalid depth sample"
            data['mask_tensor_list'].append(valid_mask_patchified)

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

            # image_tensor = self.depth_transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            raise ValueError("doesn't support depth gt map for vit")
            # vit_image_tensor = self.depth_transform(image)
            # height, width = vit_image_tensor.shape[1:]
            # data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            # data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
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
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
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


class DepthIterableDatasetWithoutMask(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    PROMPT_TEMPLATE = [
        "<depth-estimation>Estimate the depth of the image and generate the depth map.</depth-estimation>"
    ]

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        depth_transform=None, dataset_given_name="", **kwargs
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
        self.depth_normalize = ScaleShiftDepthNormalizer(norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02)
        self.depth_min = 1e-5
        self.dataset_given_name = dataset_given_name
        self.normalize_transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )
        self.include_vit_feature = kwargs.get('include_vit_feature', True)
        logger.info("#"*10 + f"include_vit_feature: {self.include_vit_feature} in {dataset_given_name}")


    def parse_row(self, row):
        
        image_data = row['images'][0]
        depth_data = row['depth'][0]
        mask_data = row['mask'][0]
        rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_data['bytes'])))
        # read depth data
        depth_raw_pil = Image.open(io.BytesIO(depth_data['bytes']))
        mask_raw_pil = Image.open(io.BytesIO(mask_data['bytes']))

        # crop transform pil
        if 'kitti' in self.dataset_given_name and self.depth_transform.kitti_bm_crop:
            rgb_image = self.depth_transform.kitti_benchmark_crop_pil(rgb_image)
            depth_raw_pil = self.depth_transform.kitti_benchmark_crop_pil(depth_raw_pil)
            mask_raw_pil = self.depth_transform.kitti_benchmark_crop_pil(mask_raw_pil)

        # aug
        if random.random() < 0.5:
            rgb_image = torchvision.transforms.functional.hflip(rgb_image)
            depth_raw_pil = torchvision.transforms.functional.hflip(depth_raw_pil)
            mask_raw_pil = torchvision.transforms.functional.hflip(mask_raw_pil)

        depth_raw = torch.from_numpy(np.asarray(depth_raw_pil).copy()).float().unsqueeze(0) # shape=(1, h, w)
        valid_mask = torch.from_numpy(np.asarray(mask_raw_pil).copy()).unsqueeze(0).bool()  # NOTE: uint16 will be 0 during resize

        # preprocessing: set the invalid depth to max plane
        min_depth, max_depth = self.get_depth_range(self.dataset_given_name)
        depth_raw = depth_raw.clamp(min=min_depth, max=max_depth)

        depth_norm = (depth_raw - min_depth) / (
            max_depth - min_depth
        )  # range [0, 1]
        depth_norm_stack = depth_norm.repeat(3, 1, 1)  # stack depth to shape=(3, h, w)
        depth_norm_vae = self.normalize_transform(depth_norm_stack)

        prompt = random.choice(self.PROMPT_TEMPLATE)
        data = self._init_data()
        # assert len(depth_norm_stack) == 1, "we only support monocular depth estimation now"
        # change the gt to channel of 4, one for mask
        # here, we need to set the vae or vit features
        data = self._add_image(data, rgb_image, need_loss=False, need_vae=True, need_vit=self.include_vit_feature)
        data = self._add_text(data, prompt, need_loss=False)
        data = self._add_depth(data, depth_norm_vae, need_loss=True, need_vae=False, need_vit=False)
        
        return data
    
    @staticmethod
    def get_depth_range(dataset):
        # depth range from Marigold
        assert dataset in ["virtualkitti_v2", "nyu", "kitti", "hypersim_train"]
        min_depth = 1e-5
        max_depth = 80.0
        if dataset == "virtualkitti_v2":
            min_depth = 1e-5
            max_depth = 80.0
        elif dataset == "nyu":
            min_depth = 1e-3
            max_depth = 10.0
        elif dataset == "kitti":
            min_depth = 1e-5
            max_depth = 80.0
        elif dataset == "hypersim_train":
            min_depth = 1e-5
            max_depth = 65.0
        return min_depth, max_depth

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
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
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_depth(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        image_tensor = image

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'need_mask': 0
                }
            )

            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)
            # downsample here
            # invalid_mask = ~valid_mask
            # valid_mask_downsample = ~torch.max_pool2d(invalid_mask.float(), self.transform.stride, self.transform.stride).bool()
            # valid_mask_downsample = valid_mask_downsample.float()
            # data['mask_tensor_list'].append(valid_mask_downsample)

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

            # image_tensor = self.depth_transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            raise ValueError("doesn't support depth gt map for vit")
            # vit_image_tensor = self.depth_transform(image)
            # height, width = vit_image_tensor.shape[1:]
            # data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            # data['image_tensor_list'].append(vit_image_tensor)

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


def visualize_depth_tensor(depth_tensor, save_path="debug.png"):

    import matplotlib.pyplot as plt

    depth_numpy = depth_tensor.cpu().squeeze().numpy()
    # depth_numpy = np.squeeze(depth_numpy, axis=0)

    cmap='viridis'
    plt.figure(figsize=(12, 5))
    plt.imshow(depth_numpy, cmap=cmap)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

