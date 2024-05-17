"""setup.py"""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.0.4"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/meehai/video-representations-extractor"

with open(f"{Path(__file__).absolute().parent}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED = ["numpy>=1.21.6", "PyYAML==6.0", "pims==0.6.1", "tqdm==4.66.1", "natsort==8.2.0", "gdown==4.6.0",
            "torch==2.2.1", "torchvision==0.17.1", "timm==0.6.13", "transforms3d==0.4.1", "pyproj>=3.2.0",
            "overrides==7.3.1", "pandas==2.1.3", "matplotlib==3.7.1", "flow_vis==0.1", "colorama==0.4.6",
            "omegaconf==2.3.0", "lovely_tensors==0.1.15", "fvcore==0.1.5.post20221221", "pycocotools==2.0.7",
            "moviepy==1.0.3", "opencv-python==4.7.0.68", "Pillow==10.3.0", "scipy==1.8.1"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    dependency_links=[],
    license="WTFPL",
    python_requires=">=3.9",
    scripts=["bin/vre", "bin/vre_collage"],
)
