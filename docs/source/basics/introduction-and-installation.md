# Introduction

The purpose of this tool is to export various representations starting from RGB videos only.Representations are defined as ways of 'looking at the world'. One can watch at various levels of information:
- low level: colors, edges
- mid level: depth, orientation of planes (normals)
- high level: semantics and actions

## Installation

### 2.1 Google Colab
Here's a recent google colab run: [link](https://colab.research.google.com/drive/1vAp71H-TLewhF56odv33TkmGwwhuoFJ-?usp=sharing)
that is based in [this examples notebook](examples/semantic_mapper/semantic_mapper.ipynb).

### 2.2 Local installation: Pip (recommended)

```
conda create -n vre python=3.11 anaconda # >=3.10 tested
pip install video-representations-extractor
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] vre example/video.mp4 --connfig_path example/cfg.yaml -o example
```

### 2.3 Local installation: Development
You can, of course, clone this repository and add it to your path for development:
```
conda create -n vre python=3.11 anaconda # >=3.10 tested
[GIT_LFS_SKIP_SMUDGE=1] git clone https://gitlab.com/video-representations-extractor/video-representations-extractor [/local/vre/dir]
pip install -e /local/vre/dir            # core deps; vendored pkg/ (vre-video, image_utils) is on the path
# Add the paths in `~/.bashrc` so it can be accessed globally from the terminal
export PYTHONPATH="$PYTHONPATH:/local/vre/dir"
export PATH="$PATH:/local/vre/dir/cli"
# Check that the installation worked
pytest /local/vre/dir/test # requires that pytest is installed
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] bash test/end_to_end/imgur/run.sh # run the e2e test
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] vre example/video.mp4 --connfig_path example/cfg.yaml -o example
```

Note: `vre-repository` (the heavy models: torch, cv2, timm, ...) is an extra: `pip install -e ".[repository]"`.

## Usage

Using the VRE CLI tool is as simple as:
```bash
vre <path/to/video.mp4> --config_path <path/to/cfg> -o <path/to/export_dir>
```

For testing that the installation works (pip or development), we run with this test video first:
```bash
mkdir example/
curl "https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/resources/test_video.mp4" \
  -o example/video.mp4 # you can of course use any video, not just our test one
curl https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/test/end_to_end/imgur/cfg.yaml -o example/cfg.yaml
```

**Single image usage**

You can get the representations for a single image (or a directory of images) by placing your image in a standalone
directory.

```bash
vre <path/to/dir_of_images> --config_path <path/to/cfg> -o <path/to/export_dir>
```

For understanding a bit more about the architecture and design of this tool, see the [architecture and design](./architecture-and-design.md) page.
