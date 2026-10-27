import os
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset
import re
import argparse
from tqdm import tqdm
import io
import base64
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


# ---------------------- 辅助函数 ----------------------

def draw_point(image, center, radius, color, thickness):
    cv2.circle(image, center, radius, color, thickness)


def pick_outline_color(image, center, sample_radius):
    x, y = center
    height, width = image.shape[:2]
    left = max(0, x - sample_radius)
    right = min(width, x + sample_radius + 1)
    top = max(0, y - sample_radius)
    bottom = min(height, y + sample_radius + 1)

    patch = image[top:bottom, left:right]
    if patch.size == 0:
        return (255, 255, 255)

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return (0, 0, 0) if patch_gray.mean() > 128 else (255, 255, 255)


def draw_visual_point(image, center, radius, color, label, outline_thickness, inner_thickness, font_scale):
    x, y = center
    outline_color = pick_outline_color(image, center, sample_radius=max(6, radius))
    outer_radius = radius + max(2, outline_thickness)

    # cv2.circle(image, (x, y), outer_radius, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, color, inner_thickness, lineType=cv2.LINE_AA)

    text_x = min(image.shape[1] - 32, x + outer_radius + 8)
    text_y = max(24, y - outer_radius - 8)
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        inner_thickness,
        cv2.LINE_AA,
    )


def draw_visual_pair(image, point, radius_scale, outline_scale, font_scale, swap=False):
    """Mark the annotated point pair, green labelled A and red labelled B.

    With swap=True the two colours change places. Every DA-2K annotation names point1 as
    the closer one, so without swapping, green marks the closer point in all 2068 items
    and the answer becomes a function of the question wording alone.
    """
    image_scale = max(image.shape[0], image.shape[1]) / 1024
    radius = max(8, int(radius_scale * image_scale))
    outline_thickness = max(2, int(outline_scale * image_scale))
    inner_thickness = max(2, outline_thickness - 1)
    green, red = ('point2', 'point1') if swap else ('point1', 'point2')
    draw_visual_point(
        image,
        (point[green][1], point[green][0]),
        radius,
        (0, 255, 0),
        label="A",
        outline_thickness=outline_thickness,
        inner_thickness=inner_thickness,
        font_scale=font_scale * image_scale,
    )
    draw_visual_point(
        image,
        (point[red][1], point[red][0]),
        radius,
        (0, 0, 255),
        label="B",
        outline_thickness=outline_thickness,
        inner_thickness=inner_thickness,
        font_scale=font_scale * image_scale,
    )


def save_debug_visualization(image, image_path_key, point_idx, debug_dir, radius_scale, outline_scale, font_scale):
    debug_dir_path = Path(debug_dir)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    debug_filename = (
        f"{Path(image_path_key).parent.name}__{Path(image_path_key).stem}"
        f"__pair{point_idx}_r{radius_scale:g}_o{outline_scale:g}_f{font_scale:g}.jpg"
    )
    debug_path = debug_dir_path / debug_filename
    cv2.imwrite(str(debug_path), image)


def encode_image_to_base64(image, extension=".png"):
    """
    直接使用 OpenCV 编码图片，避免 PIL 保存带来的额外开销。
    """
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def extract_single_choice_with_word_boundary(pred):
    """
    从预测文本中提取选项。
    """
    if pred is None:
        return None
    try:
        pred = str(pred)
    except Exception:
        return None

    pattern_1 = r'``([^`]*)``'
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r'`([^`]*)`'
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r'\b[A-F]\b(?!\s[a-zA-Z])'
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None

    return pred

# ---------------------- Dataset 定义 ----------------------

class DA2KVal(Dataset):
    def __init__(self, annfile, imagedir):
        super().__init__()
        anns = json.load(open(annfile))
        self.anns = anns
        self.anns_list = list(anns.keys())
        self.imagedir = imagedir

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        image_path_key = self.anns_list[index]
        points = self.anns[image_path_key]

        # prepare image
        full_image_path = os.path.join(self.imagedir, image_path_key)
        image = cv2.imread(full_image_path)

        prepared_samples = []
        for point_idx, point in enumerate(points):
            clone = image.copy()
            draw_visual_pair(clone, point, radius_scale=10.0, outline_scale=3.0, font_scale=0.9)
            closer_point = point['closer_point']

            rgb_img = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_img)

            # circular
            prompt = "Which point is closer to the camera: the green point or the red point?"
            prompt = prompt + "\n" + "Options:\nA. the green point\nB. the red point"
            prompt += "\nPlease select the correct answer from the options above."
            prompt = prompt.strip()
            answer = 'A' if '1' in closer_point else 'B'
            
            image_path = os.path.join(str(Path(full_image_path).parent), str(Path(full_image_path).stem) + f"_{point_idx}" + str(Path(full_image_path).suffix))
            prepared_samples.append(dict(
                        id=index,
                        image_path=image_path,
                        image=image_pil.copy(),
                        prompt=prompt,
                        answer=answer,
                        A="the green point",
                        B="the red point",
                    ))
            
            # 
            prompt = "Which point is farther from the camera: the green point or the red point?"
            prompt = prompt + "\n" + "Options:\nA. the green point\nB. the red point"
            prompt += "\nPlease select the correct answer from the options above."
            prompt = prompt.strip()
            answer = 'B' if '1' in closer_point else 'A'
            image_path = os.path.join(str(Path(full_image_path).parent), str(Path(full_image_path).stem) + f"_{point_idx}_far" + str(Path(full_image_path).suffix))
            prepared_samples.append(dict(
                        id=index,
                        image_path=image_path,
                        image=image_pil.copy(),
                        prompt=prompt,
                        answer=answer,
                        A="the green point",
                        B="the red point",
                    ))
            
        return prepared_samples


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将 PIL.Image 对象转换为 Base64 字符串。
    :param image: PIL.Image.Image 对象。
    :param format: 图像的格式，如 "PNG", "JPEG", "GIF" 等。
                   对于需要透明度的图片，推荐使用 "PNG"。
    :return: Base64 编码的字符串。
    """
    # 1. 创建一个内存中的字节流对象
    buffer = io.BytesIO()
    # 2. 将 PIL Image 对象以指定格式保存到内存流中
    #    对于 JPEG，需要处理 'P' 或 'PA' 模式的图像
    if format.upper() == 'JPEG' and image.mode in ('P', 'PA'):
        image = image.convert('RGB')
    image.save(buffer, format=format)
    # 3. 获取内存流中的全部二进制内容
    img_bytes = buffer.getvalue()
    # 4. 使用 base64 模块进行编码，并解码为 utf-8 字符串
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str


def build_samples_for_image(task):
    """
    每个进程独立处理一张原图，复用一次 `imread` 结果，返回该图对应的全部样本。
    """
    index, image_path_key, points, imagedir, config = task

    full_image_path = os.path.join(imagedir, image_path_key)
    image = cv2.imread(full_image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {full_image_path}")

    image_parent = str(Path(full_image_path).parent)
    image_stem = str(Path(full_image_path).stem)
    image_suffix = str(Path(full_image_path).suffix)

    options = "Options:\nA. the green point\nB. the red point"
    instruction = "Please select the correct answer from the options above."
    QUESTIONS = {
        'close': "Which point is closer to the camera: the green point or the red point?",
        'far': "Which point is farther from the camera: the green point or the red point?",
    }

    prepared_samples = []
    for point_idx, point in enumerate(points):
        closer_point = point['closer_point']
        closer_is_point1 = '1' in closer_point

        for swap in config['swaps']:
            clone = image.copy()
            draw_visual_pair(
                clone,
                point,
                radius_scale=config['marker_radius_scale'],
                outline_scale=config['marker_outline_scale'],
                font_scale=config['marker_font_scale'],
                swap=swap,
            )

            if config['debug']:
                save_debug_visualization(
                    clone,
                    image_path_key,
                    f'{point_idx}_swap{int(swap)}',
                    config['debug_dir'],
                    config['marker_radius_scale'],
                    config['marker_outline_scale'],
                    config['marker_font_scale'],
                )

            encoded_image = encode_image_to_base64(clone)
            # Green marks point1 unless swapped, so green is the closer point exactly when
            # the closer point is the one it currently marks.
            green_is_closer = closer_is_point1 != swap

            for polarity in config['polarities']:
                question = QUESTIONS[polarity]
                green_is_answer = green_is_closer if polarity == 'close' else not green_is_closer
                tag = f"{point_idx}_{'swap' if swap else 'orig'}_{polarity}"
                image_path = os.path.join(image_parent, f"{image_stem}_{tag}{image_suffix}")
                prepared_samples.append({
                    'question_id': f'{index}_{point_idx}',
                    'image': encoded_image,
                    'question': question,
                    'prompt': f"{question}\n{options}\n{instruction}",
                    'answer': 'A' if green_is_answer else 'B',
                    'image_path': str(Path(image_path_key).parent / f"{image_stem}_{tag}{image_suffix}"),
                    'A': "the green point",
                    'B': "the red point",
                })

    return prepared_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DA-2K dataset TSV with multiprocessing.")
    parser.add_argument(
        "--datadir",
        required=True,
        help="DA-2K dataset root directory.",
    )
    parser.add_argument(
        "--output",
        default="outputs/da2k/DA2KDataset_circular.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--axes",
        default="colour",
        choices=["colour", "colour+polarity", "polarity"],
        help="Which permutations to emit per annotated point pair. 'colour' swaps which "
             "point is green (2 rows, same cost as the original set) and is what breaks "
             "the shortcut, since every annotation makes point1 the closer one. Adding "
             "'polarity' also asks the closer/farther form of the question (4 rows).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of worker processes used for image processing.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Chunksize passed to ProcessPoolExecutor.map.",
    )
    parser.add_argument(
        "--marker-radius-scale",
        type=float,
        default=10.0,
        help="Marker radius in pixels is computed as scale * image_width / 1024.",
    )
    parser.add_argument(
        "--marker-outline-scale",
        type=float,
        default=3.0,
        help="Marker outline thickness in pixels is computed as scale * image_width / 1024.",
    )
    parser.add_argument(
        "--marker-font-scale",
        type=float,
        default=0.9,
        help="Label font scale relative to image width.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save a small set of rendered debug images for marker inspection.",
    )
    parser.add_argument(
        "--debug-dir",
        default="/tmp/da2k_debug",
        help="Directory used to save debug visualization images.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=20,
        help="Only the first N source images are saved in debug mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If positive, only process the first N source images. Useful for debug runs.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    datadir = args.datadir
    annfile = os.path.join(datadir, 'annotations.json')
    result = []
    anns = json.load(open(annfile))
    if args.debug:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    ann_items = list(anns.items())
    if args.max_images > 0:
        ann_items = ann_items[:args.max_images]

    swaps = [False, True] if 'colour' in args.axes else [False]
    polarities = ['close', 'far'] if 'polarity' in args.axes else ['close']

    for index, (image_path_key, points) in enumerate(ann_items):
        config = {
            'marker_radius_scale': args.marker_radius_scale,
            'marker_outline_scale': args.marker_outline_scale,
            'marker_font_scale': args.marker_font_scale,
            'debug': args.debug and index < args.debug_limit,
            'debug_dir': args.debug_dir,
            'swaps': swaps,
            'polarities': polarities,
        }
        tasks.append((index, image_path_key, points, datadir, config))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for sample_batch in tqdm(
            executor.map(build_samples_for_image, tasks, chunksize=args.chunksize),
            total=len(tasks),
        ):
            result.extend(sample_batch)
    
    # Index as "<question_id>_<permutation_id>" so the evaluator can group the
    # permutations of a question together and require all of them to be correct.
    qids = {}
    perm_counts = defaultdict(int)
    for sample in result:
        qkey = sample.pop('question_id')
        qid = qids.setdefault(qkey, len(qids))
        sample['index'] = f'{qid}_{perm_counts[qid]}'
        perm_counts[qid] += 1

    df = pd.DataFrame(result)
    cols = ['index'] + [c for c in df.columns if c != 'index']
    print(f"saving {len(result)} samples over {len(qids)} questions "
          f"({len(swaps)} colour x {len(polarities)} polarity permutations)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.output, sep='\t', index=False)
