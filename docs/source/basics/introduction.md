# Introduction

The purpose of this tool is to export various representations starting from RGB videos only.Representations are defined as ways of 'looking at the world'. One can watch at various levels of information:
- low level: colors, edges
- mid level: depth, orientation of planes (normals)
- high level: semantics and actions

## Installation

### 2.1 Google Colab
Here's is a recent google colab run: [link](https://colab.research.google.com/drive/1vAp71H-TLewhF56odv33TkmGwwhuoFJ-?usp=sharing)
that is based in [this examples notebook](examples/semantic_mapper/semantic_mapper.ipynb).

### 2.2 Local installation: Pip (recommended)

```
conda create -n vre python=3.11 anaconda # >=3.10 tested
pip install video-representations-extractor
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] vre example/video.mp4 --connfig_path example/cfg.yaml -o example
```

### 2.3 Local installation: Docker
We offer a pre-pushed VRE image in dockerhub. Example below:

```bash
docker run -v `pwd`/example:/app/example -v `pwd`/resources/weights:/app/weights \
  --gpus all -e VRE_DEVICE='cuda' -e VRE_WEIGHTS_DIR=/app/weights \
  meehai/vre:latest /app/example/video.mp4 \
  --config_path /app/example/cfg.yaml -o /app/example/output_dir --start_frame 100 --end_frame 101
```

Note: For the `--gpus all -e VRE_DEVICE='cuda'` part to work, you need to install `nvidia-container-toolkit` as well.
Check NVIDIA's documentation for this. If you are only on a CPU machine, then remove them from the docker run command.

### 2.4 Local installation: Development
You can, of course, clone this repository and add it to your path for development:
```
conda create -n vre python=3.11 anaconda # >=3.10 tested
[GIT_LFS_SKIP_SMUDGE=1] git clone https://gitlab.com/video-representations-extractor/video-representations-extractor [/local/vre/dir]
pip install -r /local/vre/dir/requirements.txt # there's a setup.py file too if you wish
# Add the paths in `~/.bashrc` so it can be accessed globally from the terminal
export PYTHONPATH="$PYTHONPATH:/local/vre/dir"
export PATH="$PATH:/local/vre/dir/cli"
# Check that the installation worked
pytest /local/vre/dir/test # requires that pytest is installed
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] bash test/end_to_end/imgur/run.sh # run the e2e test
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] vre example/video.mp4 --connfig_path example/cfg.yaml -o example
```

## Usage

Using the VRE CLI tool is as simple as:
```bash
vre <path/to/video.mp4> --config_path <path/to/cfg> -o <path/to/export_dir>
```

For testing that the installation works (pip or development), we run with this test video first:
```bash
mkdir example/
chmod 777 -R example/ # required only for docker
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
