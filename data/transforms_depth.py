"""
copy the depth norm code from marigold
"""
import random
from PIL import Image

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from loguru import logger

from .transforms import MaxLongEdgeMinShortEdgeResize
from .depth_encoding.power_transform import DepthPowerTransform
from .depth_encoding.multiscale_power_transform import DepthMultiScalePowerTransform
from .depth_encoding.disparity_transform import DepthDisparityTransform
from .depth_encoding.multiscale_disparity_transform import DepthMultiScaleDisparityTransform
from .depth_encoding.scale_shift_transform import ScaleShiftDepthNormalizer


class DepthTransform:
    def __init__(
        self, 
        max_image_size, 
        min_image_size, 
        image_stride, 
        max_pixels=14*14*9*1024,
        kitti_bm_crop=False,
        **kwargs
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, 
            min_size=min_image_size, 
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()

        self.kitti_bm_crop = kitti_bm_crop

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        if isinstance(img, Image.Image) or isinstance(img, np.ndarray):
            img = self.to_tensor_transform(img)
        return img

    @staticmethod
    def kitti_benchmark_crop_tensor(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor:Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out


    @staticmethod
    def kitti_benchmark_crop_pil(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (PIL.Image): Input image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        width, height = input_img.size
        top_margin = height - KB_CROP_HEIGHT
        left_margin = (width - KB_CROP_WIDTH) // 2
        right_margin = left_margin + KB_CROP_WIDTH
        bottom_margin = top_margin + KB_CROP_HEIGHT
        out = input_img.crop((left_margin, top_margin, right_margin, bottom_margin))
        return out


class DepthTransformBase:
    def __init__(
        self, 
        max_image_size, 
        min_image_size, 
        image_stride, 
        max_pixels=14*14*9*1024,
        kitti_bm_crop=False,
        normalize=None,
        stack=False,
        **kwargs
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, 
            min_size=min_image_size, 
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.kitti_bm_crop = kitti_bm_crop
        self.normalize = normalize
        self.stack = stack

    def __call__(self, img, img_num=1, valid_mask=None):
        """
        image: shape=shape=(1, h, w), tensor
        """
        # convert to tensor
        if isinstance(img, Image.Image):
            img = np.asarray(img)  # float 32 format
        if isinstance(img, np.ndarray):
            img = self.to_tensor_transform(img)
        # resize
        img = self.resize_transform(img, img_num=img_num)
        # normalize
        if self.normalize is not None:
            img = self.normalize(img, valid_mask=valid_mask)
        # copy
        if self.stack:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def kitti_benchmark_crop_tensor(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor:Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out

    @staticmethod
    def kitti_benchmark_crop_pil(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (PIL.Image): Input image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        width, height = input_img.size
        top_margin = height - KB_CROP_HEIGHT
        left_margin = (width - KB_CROP_WIDTH) // 2
        right_margin = left_margin + KB_CROP_WIDTH
        bottom_margin = top_margin + KB_CROP_HEIGHT
        out = input_img.crop((left_margin, top_margin, right_margin, bottom_margin))
        return out


def build_depth_transform(depth_transform_args):
    depth_normalization = depth_transform_args.get('depth_normalization', {'type': 'scale_shift'})
    if depth_normalization['type'] == 'scale_shift':
        logger.info(f"using scale shift transform")
        normalizer = ScaleShiftDepthNormalizer(norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02)
        depth_transform = DepthTransformBase(
            **depth_transform_args,
            normalize=normalizer,
            stack=True,
            )
    elif depth_normalization['type'] == "power_transform":
        logger.info(f"using power transform")
        normalizer = DepthPowerTransform(
            lam=depth_normalization.get('lam', -3.0), 
            c=depth_normalization.get('c', 10.0 / 3.0)
            )
        depth_transform = DepthTransformBase(
            **depth_transform_args,
            normalize=normalizer,
            stack=True,
            )
    elif depth_normalization['type'] == "multiscale_power_transform":
        logger.info(f"using power transform")
        normalizer = DepthMultiScalePowerTransform(
            lam=depth_normalization.get('lam', -3.0), 
            c=depth_normalization.get('c', [2.0, 6.0, 18.0])
            )
        depth_transform = DepthTransformBase(
            **depth_transform_args,
            normalize=normalizer,
            stack=True,
            )
    elif depth_normalization['type'] == "disparity_transform":
        logger.info(f"using disparity transform")
        normalizer = DepthDisparityTransform(
            c=depth_normalization.get('c', 5.0)
            )
        depth_transform = DepthTransformBase(
            **depth_transform_args,
            normalize=normalizer,
            stack=True,
            )
    elif depth_normalization['type'] == "multiscale_disparity_transform":
        logger.info(f"using multiscale disparity transform")
        normalizer = DepthMultiScaleDisparityTransform(
            c_values=depth_normalization.get('c', [2.0, 8.0, 32.0])
            )
        depth_transform = DepthTransformBase(
            **depth_transform_args,
            normalize=normalizer,
            stack=False,
            )
    else:
        raise ValueError(f"depth_normalization type {depth_normalization['type']} not supported")
    return depth_transform


def decode_depth_wrap(depth_transform: DepthTransformBase, depth_vae_out):
    if (depth_transform is None) or isinstance(depth_transform.normalize, ScaleShiftDepthNormalizer):
        depth_map = (depth_vae_out * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).mean(dim=-1) # (h, w)
    elif isinstance(depth_transform.normalize, DepthMultiScaleDisparityTransform) or isinstance(depth_transform.normalize, DepthMultiScalePowerTransform):
        # input shape=(3, h, w), output shape=(h, w)
        depth_map = depth_transform.normalize.decode(depth_vae_out.clamp(-1.0, 1.0)) 
    elif isinstance(depth_transform.normalize, DepthDisparityTransform) or isinstance(depth_transform.normalize, DepthPowerTransform):
        depth_map = depth_transform.normalize.decode(depth_vae_out.clamp(-1.0, 1.0).mean(dim=0))
    else:
        raise ValueError(f"Unknown depth transform type: {depth_transform}")
    return depth_map
