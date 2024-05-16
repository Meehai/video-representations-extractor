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

Weights repository for supported pretrained neural-network based representations is
[here](https://drive.google.com/drive/folders/1bWKEAiTXDpgaY2YOAFBvMqqyOGSafoIm?usp=sharing).

## 2. Usage

Installation is as easy as:
```
pip install video-representations-extractor
```

You can however, clone this repository and add it to your paths:
```
git clone https://gitlab.com/meehai/video-representations-extractor [/some/dir]
# in .bashrc
export PYTHONPATH="$PYTHONPATH:/some/dir"
export PATH="$PATH:/some/dir/bin"
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

Note: use `--cfg_path resources/cfgs/testCfg_ootb.yaml` for 'out of the box' working representations.

Note2: Use `VRE_DEVICE=cuda vre...` to use cuda. For some representations, this speeds up the process by a lot.

## 3. CFG file

The config file will have the hyperparameters required to instantiate each supported method as well as global
hyperparameters for the output. These parameters are sent to the constructor of each representation, so one can pass
additional semantics to each representation, such as classes of a semantic segmentation or the maximum global depth
value in meters.

High level format:

```
name of representation:
  type: some high level type (such as depth, semantic, edges, etc.)
  name: the implemented method's name (i.e. dexined, dpt, odoflow etc.)
  dependencies: [a list of dependencies given by their names]
  parameters: # as defined in the constructor of the implementation
    param1: value1
    param2: value2
  vre_parameters: # also known as runtime parameters (post constructor). Calls repr.vre_setup()
    device: "cuda" # for representations that have in their vre_setup() method a model.to(device) call

name of representation 2:
  type: some other type
  name: some other method
  dependencies: [name of representation] # since this representation depends on the prev one, it'll be computed after
  parameters: []
```

Example cfg file: See [out of the box supported representations](resources/cfgs/testCfg_ootb.yaml) and the CFG defined
in the [CI process](test/end_to_end/imgur/run.sh) for an actual export that is done at every commit on a real video.

Note: If the topological sort fails (because of cycle dependencies), an error will be thrown.

## 4. Output format

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

## 4.1 Collages

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

### 5. Run in docker

We don't offer a pre-pushed vre image in dockerhub. You need to build it from vre-ci (used in CI duh):

```
git clone https://gitlab.com/meehai/video-representations-extractor
cd video-representations-extractor
docker build . -f Dockerfile_run -t vre
mkdir example/
gdown https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk -O example/video.mp4 # you can use your video
cp resources/cfgs/testCfg_ootb.yaml example/cfg.yaml
docker container run -v `pwd`/example:/app/resources vre /app/resources/video.mp4 \
  --cfg_path /app/resources/cfg.yaml -o /app/resources/output_dir --start_frame 100 --end_frame 101
```
