# Video Representation Extractor

![logo](logo.png)

## 1. Description

The purpose of this repository is to export various representations starting from RGB images only. Representationds are
defined as ways of looking at the world.

For GitHub users: this is a mirror of the
[gitlab repository](https://gitlab.com/meehai/video-representations-extractor).

<u>Supported representations</u>

-   See [here](vre/representations/get_representation.py) for a comprehensive list, since it updates faster
than this README.

Weights repository for supported pretrained neural-network based representations is
[here](https://drive.google.com/drive/folders/1bWKEAiTXDpgaY2YOAFBvMqqyOGSafoIm?usp=sharing).

## 2. Usage

Add `bin/` directory to your `PATH` env variable to be able to access the `vre` tool directly from cli.

```bash
vre <path/to/video.mp4> --cfg_path <path/to/cfg> -o <path/to/vre/export/dir>
```

The magic happens inside the config file, where we define *what* representations to extract and *what* parameters are
used to instantiate said representations.

## 3. CFG file

The config file will have the hyperparameters required to instantiate each supported method as well as global
hyperparameters for the output. This means that if a depth method is pre-traied for 0-300m, this information will be
encoded in the CFG file and passed to the constructor of that particular depth method. There are also export level
parameters, such as the output resolution of the representations.

High level format:

```
name of representation:
  type: some high level type (such as depth, semantic, edges, etc.)
  method: the implemented method
  dependencies: [a list of dependencies given by their names] 
  parameters: (as defined in the constructor of the implementation)
    param1: value1
    param2: value2

name of representation 2:
  type: some other type
  method: some other method
  dependencies: [name of representation]
  parameters: []
```

Example cfg file: See [out of the box supported representations](cfgs/testCfg_ootb.yaml) and the CFG defined in
the [CI process](.gitlab-ci.yml) for an actual export that is done at every commit on a real video.

Note: If the topological sort fails (because cycle dependencies), an error will be thrown.

Note2: dependencies are provided by names and apply only to the case where one representation (say odo flow) depends
on a generic secondary representation. In this case *any* optical flow would work as long as we have a motion field
vector for each frame returned by the required dependency. In cases where dependencies can be infered automatically,
this is done behind the scenes. All representations require RGB, for example, but this is expected, so we don't need
to specify it.

## 4. Output format

All the outputs are going to be stored as [0-1] float32 npz files, one for each frame in a directory specified by
`--output_dir/-o`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:

```
/path/to/output_dir/
  name of representation/
    1.npz, ..., N.npz + cfg.yaml
  name of representation 2/
    1.npz, ..., N.npz + cfg.yaml
```

The `cfg.yaml` file for each representation is created so that we know what parameters were used for that
representation.

### 4.1 Exporting PNG images

We can also export images as a grid collage of all exported representations by adding the CLI argument
`--export_collage` to the `vre` tool.

This yields a new directory with PNGs:

```
/path/to/output_dir/
  1.npz, ..., N.npz
  png/
    1.png, ..., N.png
```

**Bonus**: Exporting video from PNGs

```bash
oldPath=`pwd`; cd /path/to/output_dir/collage; ffmpeg -start_number 1 -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p $oldPath/collage.mp4; cd -;
```

**Run in docker**
- use `meehai/vre:latest` from docker hub.

```
mkdir /tmp/example
# move the cfg and the video in some local dir
gdown https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk -O /tmp/example/vid.mp4
cp test/end_to_end/imgur/cfg.yaml /tmp/example
docker run \
  -v /tmp/example:/mnt \
  -v `pwd`/weights:/app/resources/weights \
  meehai/vre \
  /mnt/vid.mp4 --cfg_path /mnt/cfg.yaml -o /mnt/result
```
