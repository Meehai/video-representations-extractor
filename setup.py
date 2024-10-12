"""setup.py"""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "video-representations-extractor"
VERSION = "1.1.0"
DESCRIPTION = "Video Representations Extractor (VRE) for computing algorithmic or neural representations of each frame."
URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"

with open(f"{Path(__file__).absolute().parent}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED = ["numpy>=1.21.6,<2.0.0", "PyYAML==6.0", "pims==0.7.0", "tqdm==4.66.5", "natsort==8.2.0",
            "torch==2.4.1", "torchvision==0.19.1", "timm==1.0.9", "pyproj>=3.2.0", "overrides==7.7.0",
            "pandas>=2.1.3,<3.0.0", "matplotlib==3.7.1", "flow_vis==0.1", "loggez==0.4.0", "omegaconf==2.3.0",
            "lovely_tensors==0.1.17", "fvcore==0.1.5.post20221221", "pycocotools==2.0.7", "moviepy==1.0.3",
            "opencv-python==4.7.0.68", "Pillow==10.3.0", "diffusers==0.30.3"]

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
    python_requires=">=3.10",
    scripts=["bin/vre", "bin/vre_collage"],
)
