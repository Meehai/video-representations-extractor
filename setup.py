"""setup.py -- note use setuptools==73.0.1; older versions fuck up the data files, newer versions include resources."""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.15.0"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"

CWD = Path(__file__).absolute().parent
with open(CWD/"README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_CORE = [
    "vre-video>=0.5.0",
    "loggez>=0.5.0",
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
    return (x.is_file() and x.suffix not in (".py", ".pyc", ".png", ".jpg", "*.md")
            and x.name != ".gitignore" and "weights" not in x.parts)
glob_files = lambda x: list(Path(x).glob("**/*")) # pylint: disable=all
vre_files = [str(x) for x in glob_files("vre/") if _filter_file(x)]
vre_repo_files = [str(x) for x in glob_files("vre_repository/") if _filter_file(x)]
data_files = [("", [*vre_files, *vre_repo_files])]

packages = find_packages()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=packages,
    data_files=data_files,
    package_data={"": data_files[0][1]},
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
