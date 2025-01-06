# pylint: disable=all
import numpy as np
import torch
from PIL import Image
from vre.logger import vre_logger as logger
import math
from types import SimpleNamespace
import re
import yaml
from pathlib import Path

class IterableSimpleNamespace(SimpleNamespace):
    """
    Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)

DEFAULT_CFG_DICT = {'task': 'detect', 'mode': 'train', 'model': None, 'data': None, 'epochs': 100, 'patience': 50, 'batch': 16, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': None, 'workers': 8, 'project': None, 'name': None, 'exist_ok': False, 'pretrained': True, 'optimizer': 'auto', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 0, 'resume': False, 'amp': True, 'fraction': 1.0, 'profile': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'show': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'vid_stride': 1, 'line_width': None, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'boxes': True, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': False, 'opset': None, 'workspace': 4, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'cfg': None, 'v5loader': False, 'tracker': 'botsort.yaml'}
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
DEFAULT_CFG_KEYS = list(DEFAULT_CFG_DICT.keys())


# Define keys for arg type checks
CFG_FLOAT_KEYS = 'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear'
CFG_FRACTION_KEYS = ('dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr',
                     'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud',
                     'fliplr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction')  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride',
                'line_width', 'workspace', 'nbs', 'save_period')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val',
                 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop',
                 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'boxes', 'keras',
                 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'v5loader', 'profile')

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg

def get_cfg(cfg=DEFAULT_CFG_DICT, overrides: dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        logger.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    # Return instance
    return IterableSimpleNamespace(**cfg)

def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    '''Adjust bounding boxes to stick to image border if they are within a certain threshold.
    Args:
    boxes: (n, 4)
    image_shape: (height, width)
    threshold: pixel threshold
    Returns:
    adjusted_boxes: adjusted bounding boxes
    '''

    # Image dimensions
    h, w = image_shape

    # Adjust boxes
    boxes[:, 0] = torch.where(boxes[:, 0] < threshold, torch.tensor(
        0, dtype=torch.float, device=boxes.device), boxes[:, 0])  # x1
    boxes[:, 1] = torch.where(boxes[:, 1] < threshold, torch.tensor(
        0, dtype=torch.float, device=boxes.device), boxes[:, 1])  # y1
    boxes[:, 2] = torch.where(boxes[:, 2] > w - threshold, torch.tensor(
        w, dtype=torch.float, device=boxes.device), boxes[:, 2])  # x2
    boxes[:, 3] = torch.where(boxes[:, 3] > h - threshold, torch.tensor(
        h, dtype=torch.float, device=boxes.device), boxes[:, 3])  # y2

    return boxes



def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def bbox_iou(box1, boxes, iou_thres=0.9, image_shape=(640, 640), raw_output=False):
    '''Compute the Intersection-Over-Union of a bounding box with respect to an array of other bounding boxes.
    Args:
    box1: (4, )
    boxes: (n, 4)
    Returns:
    high_iou_indices: Indices of boxes with IoU > thres
    '''
    boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    # obtain coordinates for intersections
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    # compute the area of intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # compute the area of both individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # compute the area of union
    union = box1_area + box2_area - intersection

    # compute the IoU
    iou = intersection / union  # Should be shape (n, )
    if raw_output:
        if iou.numel() == 0:
            return 0
        return iou

    # get indices of boxes with IoU > thres
    high_iou_indices = torch.nonzero(iou > iou_thres).flatten()

    return high_iou_indices


def image_to_np_ndarray(image):
    if type(image) is str:
        return np.array(Image.open(image))
    elif issubclass(type(image), Image.Image):
        return np.array(image)
    elif type(image) is np.ndarray:
        return image
    return None


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
                        f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list " \
              "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        logger.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        logger.warning(f'WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz
