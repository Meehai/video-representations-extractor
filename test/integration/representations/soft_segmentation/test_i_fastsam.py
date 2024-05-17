from types import SimpleNamespace
from pathlib import Path
from vre.representations.soft_segmentation.fastsam.fastsam import main as fastsam_main
from vre.utils import get_project_root
from tempfile import NamedTemporaryFile
import random

def test_i_mask2former():
    variants = ["fastsam-x", "fastsam-s"]
    args = SimpleNamespace(
        model_id=random.choice(variants),
        input_image=get_project_root() / "resources/demo1.jpg",
        output_path=Path(NamedTemporaryFile(suffix=".jpg").name),
    )
    fastsam_main(args)
