"""setup.py -- note use setuptools==73.0.1; older versions fuck up the data files, newer versions include resources."""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.18.0"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"

CWD = Path(__file__).absolute().parent
for _submodule in ("vre-video/vre_video", "image_utils"):
    assert (CWD / _submodule / "__init__.py").exists(), \
        f"Submodule '{_submodule}' not found. Run: git submodule update --init --recursive"
with open(CWD/"README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_CORE = [
    "loggez>=0.8.4",
    "numpy>=1.21.6",
    "PyYAML==6.0.3",
    "tqdm==4.66.5",
    "overrides==7.7.0",
    "Pillow==11.3.0",
    "graphviz==0.20.3",
]

REQUIRED_REPOSITORY = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "opencv-python==4.12.0.88",
    "pycocotools==2.0.10",
    "timm==1.0.9",
    "diffusers==0.30.3",
]

def _filter_file(x: Path) -> bool:
    return (x.is_file() and x.suffix not in (".py", ".pyc", ".png", ".jpg", ".md")
            and x.name != ".gitignore" and "weights" not in x.parts)
glob_files = lambda x: list(Path(x).glob("**/*")) # pylint: disable=all

packages = find_packages() + find_packages(where="vre-video")

def _build_package_data(roots: list[str]) -> dict[str, list[str]]:
    """Ship non-py data files (e.g. marigold's empty_text_embed.pkl, mask2former's *.json) as package_data
    keyed by their owning package, so they install *into* site-packages next to the code where
    `Path(__file__).parent / ...` can find them. data_files installs relative to sys.prefix instead, which
    flattens the paths and breaks those runtime lookups."""
    pkg_dirs = {p: Path(p.replace(".", "/")) for p in find_packages()}
    result: dict[str, list[str]] = {}
    for root in roots:
        for f in glob_files(root):
            if not _filter_file(f):
                continue
            owners = [(p, d) for p, d in pkg_dirs.items() if d in f.parents]
            assert owners, f"data file '{f}' is not inside any package; it won't be importable at runtime"
            pkg, pkg_dir = max(owners, key=lambda pd: len(pd[1].parts)) # deepest enclosing package
            result.setdefault(pkg, []).append(str(f.relative_to(pkg_dir)))
    return result

package_data = _build_package_data(["vre/", "vre_repository/"])

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=packages,
    package_dir={"vre_video": "vre-video/vre_video"},
    package_data=package_data,
    include_package_data=False,
    install_requires=REQUIRED_CORE,
    extras_require={
        "core": [],
        "repository": REQUIRED_REPOSITORY,
    },
    dependency_links=[],
    license="MIT",
    python_requires=">=3.10",
    scripts=["cli/vre", "cli/vre_collage", "cli/vre_reader", "cli/vre_gpu_parallel", "cli/vre_dir_analysis"],
)
