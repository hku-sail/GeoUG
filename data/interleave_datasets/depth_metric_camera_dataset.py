import json
import math
import random

from PIL import Image
from torchvision.transforms import InterpolationMode

from ..transforms import MaxLongEdgeMinShortEdgeResize
from .depth_metric_dataset import MetricDepthIterableDataset


# For sources whose meta_info has no camera_intrinsics; values are in the stored image resolution.
FALLBACK_INTRINSICS = {
    # Official calibration at 2048x1024 (fx varies < 0.3% across sequences).
    'Cityscapes': {'fx': 2262.52, 'fy': 2265.30, 'cx': 1096.98, 'cy': 513.137},
    # Official 1242x375 intrinsics shifted by the stored KITTI benchmark crop (top 23, left 13) -> 1216x352.
    'virtualkitti': {'fx': 725.0087, 'fy': 725.0087, 'cx': 607.5, 'cy': 164.0},
}

# |fx / fy - 1| above this means non-square pixels (e.g. AgiBot head camera: 1280x720 squeezed to 640x480).
NON_SQUARE_PIXEL_TOL = 0.05


class MetricDepthCameraPromptIterableDataset(MetricDepthIterableDataset):
    """MetricDepthIterableDataset that states the horizontal FOV in the prompt with prob `camera_prompt_prob`.

    Images with non-square pixels are first stretched horizontally so that fx == fy.
    Horizontal FOV is invariant to resizing and horizontal flipping, and only needs fx.
    The depth map is resized with nearest neighbour instead of bicubic.
    """

    def __init__(self, *args, camera_prompt_prob=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_prompt_prob = camera_prompt_prob
        # Bicubic invents depths between foreground and background (and negatives) at depth edges
        # and next to masked-out zeros; same output size, pixel-centre aligned like the RGB resize.
        resize = self.depth_transform.resize_transform
        self.depth_transform.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=resize.max_size, min_size=resize.min_size, stride=resize.stride,
            max_pixels=resize.max_pixels, interpolation=InterpolationMode.NEAREST_EXACT,
        )

    @staticmethod
    def _intrinsics(row):
        meta = json.loads(row['meta_info'])
        return meta.get('camera_intrinsics') or FALLBACK_INTRINSICS.get(meta.get('source'))

    @staticmethod
    def _is_non_square(intrinsics):
        return 'fy' in intrinsics and abs(intrinsics['fx'] / intrinsics['fy'] - 1) > NON_SQUARE_PIXEL_TOL

    def _preprocess_pil(self, row, rgb_image, depth_pil, mask_pil):
        intrinsics = self._intrinsics(row)
        if intrinsics is None or not self._is_non_square(intrinsics):
            return rgb_image, depth_pil, mask_pil
        # z-depth values are unchanged by the stretch; nearest keeps depth / mask edges unmixed.
        width, height = rgb_image.size
        size = (round(width * intrinsics['fy'] / intrinsics['fx']), height)
        rgb_image = rgb_image.resize(size, Image.BILINEAR)
        depth_pil = depth_pil.resize(size, Image.NEAREST)
        if mask_pil is not None:
            mask_pil = mask_pil.resize(size, Image.NEAREST)
        return rgb_image, depth_pil, mask_pil

    def _build_prompt(self, row, rgb_image):
        prompt = super()._build_prompt(row, rgb_image)
        if random.random() >= self.camera_prompt_prob:
            return prompt
        intrinsics = self._intrinsics(row)
        if intrinsics is None:
            return prompt
        # after _preprocess_pil, a non-square-pixel image has fx == fy
        fx = intrinsics['fy'] if self._is_non_square(intrinsics) else intrinsics['fx']
        hfov = math.degrees(2 * math.atan(rgb_image.width / (2 * fx)))
        body = prompt[:-len('</GENERATE_DEPTH>')].rstrip()
        if not body.endswith('.'):
            body += '.'
        return f'{body} The horizontal field of view is {hfov:.1f} degrees.</GENERATE_DEPTH>'
