import io
import random
import json
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..constant import IMAGE_FLAG, INTERLEAVE_COT_PROMPT, DEPTH_FLAG

from ..depth_encoding.scale_shift_transform import ScaleShiftDepthNormalizer

import torch
import numpy as np

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

from .camera_prompt_templates import CAMERA_PROMPT_TEMPLATES
LONG_CAMERA_PROMPT_PROB = 0.3
ACTION_STYLE_MOTION_PROB = 0.5


class MultiViewIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def change_format(self, data, num_images, image_input_list):
        elements = []
        image_idx = 0
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                has_loss  = 0
                need_vae = True
                need_vit = True
            else:
                has_loss = 1
                need_vae = False
                need_vit = False
            if IMAGE_FLAG not in conversation['value']:
                elements.append({
                    'type': 'text',
                    'has_loss': has_loss,
                    'text': conversation['value'],
                })
            else:
                text_list = conversation['value'].split(IMAGE_FLAG)
                for idx, text in enumerate(text_list):
                    if text.strip() != '':
                        elements.append({
                            'type': 'text',
                            'has_loss': has_loss,
                            'text': text.strip(),
                        })
                    if (idx != len(text_list) - 1) and (idx < num_images):
                        # depth map 过 vae，不过 vit
                        # rgb 同时过 vit 和 vae
                        # 只有理解数据，最好也过 vae，因为生成之后还有理解，这里是 非 diffusion forcing 的方式
                        elements.append({
                            'type': 'image', 
                            'has_loss': has_loss, 
                            'image': image_input_list[image_idx]['image'], 
                            'need_vae': need_vae,
                            'need_vit': need_vit,
                            })
                        image_idx += 1
        return elements

    def parse_row(self, row):

        image_info_list = row['images']
        # load depth and image, and transform
        image_input_list = []
        for image_info in image_info_list:
            image_type = 'rgb'
            rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_info['bytes'])))
            image_input_list.append({'image': rgb_image, 'type': image_type})

        conv_value  = "".join([conv['value'] for conv in row['conversations']])
        assert conv_value.count(IMAGE_FLAG) == len(image_input_list)
        
        # change convs
        elements = self.change_format(row, len(image_input_list), image_input_list)
        # prepare data
        data = self._init_data()
        # convs
        for item in elements:
            if item['type'] == 'text':
                data = self._add_text(data, item['text'], need_loss=item['has_loss'], enable_cfg=False)
            elif item['type'] == 'image':
                data = self._add_image(data, item['image'], need_loss=item['has_loss'], need_vae=item['need_vae'], need_vit=item['need_vit'])
        has_loss = [item['loss'] for item in data['sequence_plan']]
        if sum(has_loss) == 0:
            print(f'No loss defined, skipped.')
            return dict()
        
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


def generate_camera_movement_text(inst_dict):
    """
    将相机运动指令字典转换为自然语言描述
    """
    # 如果包含 Stay 或者字典为空，直接返回保持静止
    if not inst_dict or "Stay" in inst_dict:
        return "Keep the camera stationary"
        
    # 位移映射 (单位: m)
    translation_map = {
        "Dolly In": "forward",
        "Dolly Out": "backward",
        "Truck Left": "left",
        "Truck Right": "right",
        "Pedestal Up": "up",
        "Pedestal Down": "down"
    }
    
    # 旋转映射 (单位: degrees)
    rotation_map = {
        "Tilt Up": "up",
        "Tilt Down": "down",
        "Pan Left": "left",
        "Pan Right": "right",
        "Roll CW": "clockwise",
        "Roll CCW": "counterclockwise"
    }
    
    move_parts = []
    rotate_parts = []
    
    # 遍历字典中的每一个动作
    for action, value in inst_dict.items():
        if action in translation_map:
            # TODO: remove this hard code
            # if float(value) > 2.0:
            #     raise ValueError(f"Translation value {value} is too large, should be less than 2.0")
            move_parts.append(f"{translation_map[action]} {value} m")
        elif action in rotation_map:
            rotate_parts.append(f"{rotation_map[action]} {value} degrees")
            
    # 组合自然语言句子
    phrases = []
    if move_parts:
        phrases.append("move the camera " + ", ".join(move_parts))
    if rotate_parts:
        phrases.append("rotate the camera " + ", ".join(rotate_parts))
        
    if not phrases:
        return "Keep the camera stationary."
        
    # 用 and 连接位移和旋转，首字母大写
    result = " and ".join(phrases)
    return result.capitalize()


def generate_camera_action_text(inst_dict, translation_types, rotation_types):
    """
    将相机运动指令字典转换为动作名风格描述，如 dolly in / pan left。
    """
    if not inst_dict:
        return "stay"

    action_parts = []
    for action, value in inst_dict.items():
        if action in translation_types:
            # TODO: remove this hard code
            # if float(value) > 2.0:
            #     raise ValueError(f"Translation value {value} is too large, should be less than 2.0")
            action_parts.append(f"{action.lower()} {value} m")
        elif action in rotation_types:
            action_parts.append(f"{action.lower()} {value} degrees")
        elif action == "Stay":
            action_parts.append("stay")

    if not action_parts:
        return "stay"

    return ", ".join(action_parts)


def build_camera_prompt(inst_dict, translation_types, rotation_types):
    if random.random() < ACTION_STYLE_MOTION_PROB:
        motion_text = generate_camera_action_text(inst_dict, translation_types, rotation_types)
    else:
        motion_text = generate_camera_movement_text(inst_dict)

    if CAMERA_PROMPT_TEMPLATES and random.random() < LONG_CAMERA_PROMPT_PROB:
        return random.choice(CAMERA_PROMPT_TEMPLATES).format(motion_text=motion_text)

    short_prompts = [
        f"Camera coordinate: +X right, +Y downward, +Z forward. Change camera: {motion_text}",
        f"Camera coordinate: +X right, +Y downward, +Z forward. {motion_text}",
        f"{motion_text}",
    ]
    return random.choice(short_prompts)


class MultiViewIterableDatasetRaw(MultiViewIterableDataset):
    """需要在这个数据集中把 instruction 转换为 text prompt"""
    translation_types = ["Dolly In", "Dolly Out", "Truck Right", "Truck Left", "Pedestal Down", "Pedestal Up"]
    rotation_types = ["Pan Left", "Pan Right", "Tilt Down", "Tilt Up", "Roll CCW", "Roll CW"]

    def parse_row(self, row):

        image_info_list = row['images']
        # load depth and image, and transform
        image_input_list = []
        for image_info in image_info_list:
            image_type = 'rgb'
            rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_info['bytes'])))
            image_input_list.append({'image': rgb_image, 'type': image_type})

        motion_values = json.loads(row['instruction'])['values']
        for k, v in motion_values.items():
            if k != "Stay" and v < 0:
                raise ValueError
        prompt = build_camera_prompt(motion_values, self.translation_types, self.rotation_types)
        
        conversations = [
            {"from": "human", "value": IMAGE_FLAG + prompt},
            {"from": "gpt", "value": IMAGE_FLAG}
        ]
        row["conversations"] = conversations

        conv_value  = "".join([conv['value'] for conv in row['conversations']])
        assert conv_value.count(IMAGE_FLAG) == len(image_input_list)
        
        # change convs
        elements = self.change_format(row, len(image_input_list), image_input_list)
        # prepare data
        data = self._init_data()
        # convs
        for item in elements:
            if item['type'] == 'text':
                data = self._add_text(data, item['text'], need_loss=item['has_loss'], enable_cfg=False)
            elif item['type'] == 'image':
                data = self._add_image(data, item['image'], need_loss=item['has_loss'], need_vae=item['need_vae'], need_vit=item['need_vit'])
        has_loss = [item['loss'] for item in data['sequence_plan']]
        if sum(has_loss) == 0:
            print(f'No loss defined, skipped.')
            return dict()
        
        return data


class MultiViewIterableDatasetDepth(MultiViewIterableDatasetRaw):
    """
    depth in context
    """
    def __init__(self, *args, depth_transform=None, dataset_given_name=None, **kwargs):
        super().__init__(*args, **kwargs)

        # default image stride is 16 (flux vae encode)
        self.depth_transform = depth_transform
        assert self.depth_transform is not None

        self.depth_min = 1e-5

    def change_format(self, data, num_images, image_input_list):
        elements = []
        image_idx = 0
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                has_loss  = 0
            else:
                has_loss = 1
            if IMAGE_FLAG not in conversation['value']:
                elements.append({
                    'type': 'text',
                    'has_loss': has_loss,
                    'text': conversation['value'],
                })
            else:
                text_list = conversation['value'].split(IMAGE_FLAG)
                for idx, text in enumerate(text_list):
                    if text.strip() != '':
                        elements.append({
                            'type': 'text',
                            'has_loss': has_loss,
                            'text': text.strip(),
                        })
                    if (idx != len(text_list) - 1) and (idx < num_images):
                        # depth map 过 vae，不过 vit
                        # rgb 同时过 vit 和 vae
                        # 只有理解数据，最好也过 vae，因为生成之后还有理解，这里是 非 diffusion forcing 的方式
                        if image_input_list[image_idx]['type'] == 'rgb':
                            elements.append({
                                'type': 'image', 
                                'has_loss': has_loss, 
                                'image': image_input_list[image_idx]['image'], 
                                'need_vae': True if has_loss == 0 else False,  # 只有生成，生成之后没有了
                                'need_vit': True if has_loss == 0 else False,
                                })
                        elif image_input_list[image_idx]['type'] == 'depth':
                            elements.append({
                                'type': 'depth',
                                'has_loss': has_loss, 
                                'depth': image_input_list[image_idx]['image'], 
                                'need_vae': True if has_loss == 0 else False,
                                'need_vit': False,
                            })
                        image_idx += 1
        return elements

    def parse_row(self, row):

        assert 'images' in row
        # load depth and image, and transform
        image_input_list = []
        for image_info in row['images']:
            image_type = 'rgb'
            rgb_image = pil_img2rgb(Image.open(io.BytesIO(image_info['bytes'])))
            image_input_list.append({'image': rgb_image, 'type': image_type})
        
        # read depth
        depth_input_list = []
        for depth_data in row['depth']:
            depth_raw_pil = Image.open(io.BytesIO(depth_data['bytes']))
            depth_norm_stack = self.depth_transform(depth_raw_pil)
            depth_input_list.append({'image': depth_norm_stack, 'type': 'depth'}) # shape = (3, h, w)

        motion_values = json.loads(row['instruction'])['values']
        for k, v in motion_values.items():
            if k != "Stay" and v < 0:
                raise ValueError
        prompt = build_camera_prompt(motion_values, self.translation_types, self.rotation_types)
        
        if random.random() < 0.8:
            # hack, our format is <image> <depth> <image>
            # if len(depth_input_list) == 0:
            #     import pdb; pdb.set_trace()
            image_input_list.insert(1, depth_input_list[0])  # HACK: insert depth in between image
            conversations = [
                {"from": "human", "value": IMAGE_FLAG + DEPTH_FLAG + IMAGE_FLAG + prompt},
                {"from": "gpt", "value": IMAGE_FLAG}
            ]
        else:
            # hack, our format is <image> <depth> <image>
            # image_input_list.insert(1, depth_input_list[0])
            conversations = [
                {"from": "human", "value": IMAGE_FLAG + prompt},
                {"from": "gpt", "value": IMAGE_FLAG}
            ]
    
        row["conversations"] = conversations

        conv_value  = "".join([conv['value'] for conv in row['conversations']])
        assert conv_value.count(IMAGE_FLAG) == len(image_input_list)
        
        # change convs
        elements = self.change_format(row, len(image_input_list), image_input_list)
        # prepare data
        data = self._init_data()
        # convs
        for item in elements:
            if item['type'] == 'text':
                data = self._add_text(data, item['text'], need_loss=item['has_loss'], enable_cfg=False)
            elif item['type'] == 'image':
                data = self._add_image(data, item['image'], need_loss=item['has_loss'], need_vae=item['need_vae'], need_vit=item['need_vit'])
            elif item['type'] == 'depth':
                data = self._add_depth(data, item['depth'], need_loss=item['has_loss'], need_vae=item['need_vae'], need_vit=item['need_vit'])

        has_loss = [item['loss'] for item in data['sequence_plan']]
        if sum(has_loss) == 0:
            print(f'No loss defined, skipped.')
            return dict()
        
        return data

    def _add_depth(self, data, image, valid_mask=None, need_loss=False, need_vae=False, need_vit=False, enable_cfg=True):

        image_tensor = image  # shape=(B, H, W)

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