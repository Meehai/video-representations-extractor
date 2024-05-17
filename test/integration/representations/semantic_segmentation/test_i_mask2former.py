from types import SimpleNamespace
from pathlib import Path
from vre.representations.semantic_segmentation.mask2former.mask2former import main as m2f_main
from vre.utils import get_project_root
from tempfile import NamedTemporaryFile
import random

def test_i_mask2former():
    variants = ["47429163_0", "49189528_1", "49189528_0"]
    args = SimpleNamespace(
        model_id_or_path=random.choice(variants),
        input_image=get_project_root() / "resources/demo1.jpg",
        output_path=Path(NamedTemporaryFile(suffix=".jpg").name),
        n_tries=1
    )
    m2f_main(args)
