"""image utils for marigold"""
import math
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

def resize_max_res(img: torch.Tensor, max_edge_resolution: int, resample_method: InterpolationMode) -> torch.Tensor:
    """Resize image to limit maximum edge length while keeping aspect ratio."""
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def find_batch_size(ensemble_size: int, input_res: int, dtype: torch.dtype) -> int:
    """Automatically search for suitable operating batch size."""

    # Search table for suggested max. inference batch size
    bs_search_table = [
        # tested on A100-PCIE-80GB
        {"res": 768, "total_vram": 79, "bs": 35, "dtype": torch.float32},
        {"res": 1024, "total_vram": 79, "bs": 20, "dtype": torch.float32},
        # tested on A100-PCIE-40GB
        {"res": 768, "total_vram": 39, "bs": 15, "dtype": torch.float32},
        {"res": 1024, "total_vram": 39, "bs": 8, "dtype": torch.float32},
        {"res": 768, "total_vram": 39, "bs": 30, "dtype": torch.float16},
        {"res": 1024, "total_vram": 39, "bs": 15, "dtype": torch.float16},
        # tested on RTX3090, RTX4090
        {"res": 512, "total_vram": 23, "bs": 20, "dtype": torch.float32},
        {"res": 768, "total_vram": 23, "bs": 7, "dtype": torch.float32},
        {"res": 1024, "total_vram": 23, "bs": 3, "dtype": torch.float32},
        {"res": 512, "total_vram": 23, "bs": 40, "dtype": torch.float16},
        {"res": 768, "total_vram": 23, "bs": 18, "dtype": torch.float16},
        {"res": 1024, "total_vram": 23, "bs": 10, "dtype": torch.float16},
        # tested on GTX1080Ti
        {"res": 512, "total_vram": 10, "bs": 5, "dtype": torch.float32},
        {"res": 768, "total_vram": 10, "bs": 2, "dtype": torch.float32},
        {"res": 512, "total_vram": 10, "bs": 10, "dtype": torch.float16},
        {"res": 768, "total_vram": 10, "bs": 5, "dtype": torch.float16},
        {"res": 1024, "total_vram": 10, "bs": 3, "dtype": torch.float16},
    ]


    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
    filtered_bs_search_table = [s for s in bs_search_table if s["dtype"] == dtype]
    for settings in sorted(filtered_bs_search_table, key=lambda k: (k["res"], -k["total_vram"])):
        if input_res <= settings["res"] and total_vram >= settings["total_vram"]:
            bs = settings["bs"]
            if bs > ensemble_size:
                bs = ensemble_size
            elif bs > math.ceil(ensemble_size / 2) and bs < ensemble_size: # pylint: disable=chained-comparison
                bs = math.ceil(ensemble_size / 2)
            return bs

    return 1
