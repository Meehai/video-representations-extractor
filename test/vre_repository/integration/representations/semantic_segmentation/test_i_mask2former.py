import os
from types import SimpleNamespace
from tempfile import NamedTemporaryFile
from pathlib import Path
import numpy as np
from vre_repository.semantic_segmentation.mask2former.mask2former import main as m2f_main
from vre.utils import get_project_root
import pytest

IN_CI = "CI" in os.environ
BIG_CKPT = pytest.mark.skipif(IN_CI, reason="~830MB checkpoint: too slow for CI, run locally")

# CI runs the smallest checkpoint (49189528_1, 169MB), the one the e2e configs use too
@pytest.mark.parametrize("variant", [pytest.param("47429163_0", marks=BIG_CKPT), "49189528_1",
                                     pytest.param("49189528_0", marks=BIG_CKPT)])
def test_i_mask2former(variant):
    args = SimpleNamespace(
        model_id=variant,
        input_image=get_project_root() / "resources/demo1.jpg",
        output_path=Path(NamedTemporaryFile(suffix=".jpg").name),
    )
    semantic_result = m2f_main(args)

    # Sanity checks
    rtol = 1e-2
    if args.model_id == "47429163_0" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 0.0084324, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 0.087728, rtol=rtol), std
    elif args.model_id == "49189528_1" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 0.0161621, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 0.118579, rtol=rtol), std
    elif args.model_id == "49189528_0" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 0.0049801, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 0.063540, rtol=rtol), std
