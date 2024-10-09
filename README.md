# Video Representation Extractor

![logo](logo.png)

## 1. Description

The purpose of this repository is to export various representations starting from RGB videos only. Representations are
defined as ways of 'looking at the world'. One can watch at various levels of information:
- low level: colors, edges
- mid level: depth, orientation of planes (normals)
- high level: semantics and actions

For GitHub users: this is a mirror of the
[gitlab repository](https://gitlab.com/meehai/video-representations-extractor).

<u>Supported representations</u>

- See [here](vre/representations/build_representations.py) for a comprehensive list, since it updates faster
than this README.

Weights are stored in this directory using `git-lfs`: [weights dir](./resources/weights/). If you just want to download
the code, without the weights resources, use `GIT_LFS_SKIP_SMUDGE=1 git clone ...`.

## 2. Usage

### Pip

Installation is as easy as:
```
conda create -n vre python=3.11 anaconda # >=py3.8 in theory, >=3.10 tested
pip install video-representations-extractor
```

### Docker
We offer a pre-pushed VRE image in dockerhub.

```
mkdir example/
chmod 777 -R example/ # optional ?
curl "https://gitlab.com/meehai/video-representations-extractor/-/raw/master/resources/test_video.mp4" \
  -o example/video.mp4 # you can of course use any video, not just our test one
curl https://gitlab.com/meehai/video-representations-extractor/-/raw/master/test/end_to_end/imgur/cfg.yaml -o example/cfg.yaml
docker run -v `pwd`/example:/app/example -v `pwd`/resources/weights:/app/weights \
  --gpus all -e VRE_DEVICE='cuda' -e VRE_WEIGHTS_DIR=/app/weights \
  meehai/vre:latest /app/example/video.mp4 \
  --cfg_path /app/example/cfg.yaml -o /app/example/output_dir --start_frame 100 --end_frame 101
```

Note: For the `--gpus all -e VRE_DEVICE='cuda'` part to work, you need to install `nvidia-container-toolkit` as well.
Check NVIDIA's documentation for this. If you are only on a CPU machine, then remove them from the docker run command.

### Development
You can, of course, clone this repository and add it to your path for development:
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://gitlab.com/meehai/video-representations-extractor [/some/dir]
# in .bashrc
export PYTHONPATH="$PYTHONPATH:/some/dir"
export PATH="$PATH:/some/dir/bin"
pytest test/
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] bash test/end_to_end/imgur/run.sh
```

After either option, you should be able to run:
```bash
vre <path/to/video.mp4> --cfg_path <path/to/cfg> -o <path/to/export_dir>
```

The magic happens inside the config file, where we define *what* representations to extract and *what* parameters are
used to instantiate said representations.

### 2.1 Single image usage

You can get the representations for a single image (or a directory of images) by placing your image in a standalone
directory.

```bash
vre <path/to/dir_of_images> --cfg_path <path/to/cfg> -o <path/to/export_dir>
```

Note: use `--cfg_path test/end_to_end/imgur/cfg.yaml` for 'out of the box' working representations.

Note2: Use `VRE_DEVICE=cuda vre...` to use cuda. For some representations, this speeds up the process by a lot.

## 3. Details about inputs and outputs

### 3.1 Video

Any video format that is supported by `pims`. Representations were mostly tested on UAV-like videos, but they should
be fine for self driving videos or even indoor handheld videos.

### 3.2 Config files

The config file will have the hyperparameters required to instantiate each supported method as well as global
hyperparameters for the output. These parameters are sent to the constructor of each representation, so one can pass
additional semantics to each representation, such as classes of a semantic segmentation or the maximum global depth
value in meters.

High level format:

```
name of representation:
  type: some high level type (such as depth/dpt, semantic/mask2former, edges/dexined etc.)
  dependencies: [a list of dependencies given by their names]
  parameters: # as defined in the constructor of the implementation
    param1: value1
    param2: value2
  device: "cuda" # for representations that have in their vre_setup() method a model.to(device) call

name of representation 2:
  type: some other type
  name: some other method
  dependencies: [name of representation] # since this representation depends on the prev one, it'll be computed after
  parameters: []
```

Example cfg file: See [out of the box supported representations](test/end_to_end/imgur/cfg.yaml) and the CFG defined
in the [CI process](test/end_to_end/imgur/run.sh) for an actual export that is done at every commit on a real video.

Note: If the topological sort fails (because of cycle dependencies), an error will be thrown.

## 3.3. Output format

All the outputs are going to be stored as [0-1] float32 npz files, one for each frame in a directory specified by
`--output_dir/-o`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:

```
/path/to/output_dir/
  name of representation/
    npy/ # if export_npy is set
      1.npz, ..., N.npz
    png/ # if export_png is set
      1.png, ..., N.png
  name of representation 2/
    npy/
      1.npz, ..., N.npz
```

The `cfg.yaml` file for each representation is created so that we know what parameters were used for that
representation.

## 3.4 Collages

In `bin/` we provide a secondary tool, `vre_collage` that takes all the png files from an output_dir as above and
stacks them together in a single image. This is useful if we want to create a single image of all representations which
can later be turned into a video as well.

Usage:
```
vre_collage /path/to/output_dir -o /path/to/collage_dir [--overwrite] [--video] [--fps] [--output_resolution H W]
```

Note: you can also get video from a collage dir like this (in case you forgot to set --video or want more control):

```bash
cd /path/to/collage_dir
ffmpeg -start_number 1 -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p /path/to/collage.mp4;
```
