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

Add `bin/` directory to your `PATH` env variable to be able to access the `vre` tool directly from cli.

```bash
vre <path/to/video.mp4> --cfg_path <path/to/cfg> -o <path/to/export_dir>
```

The magic happens inside the config file, where we define *what* representations to extract and *what* parameters are
used to instantiate said representations.

### Single image usage

You can get the representations for a single image (or a directory of images) by placing your image in a standalone
directory.

```bash
vre <path/to/dir_of_images> --cfg_path <path/to/cfg> -o <path/to/export_dir>
```

Note: use `--cfg_path resources/cfgs/testCfg_ootb.yaml` for 'out of the box' working representations.
Note2: Use `VRE_DEVICE=cuda vre...` to use cuda. For some representations, this speeds up the process by a lot.

## 3. CFG file

The config file will have the hyperparameters required to instantiate each supported method as well as global
hyperparameters for the output. This means that if a depth method is pre-traied for 0-300m, this information will be
encoded in the CFG file and passed to the constructor of that particular depth method. There are also export level
parameters, such as the output resolution of the representations.

High level format:

```
name of representation:
  type: some high level type (such as depth, semantic, edges, etc.)
  name: the implemented method's name (i.e. dexined, dpt, odoflow etc.)
  dependencies: [a list of dependencies given by their names]
  parameters: (as defined in the constructor of the implementation)
    param1: value1
    param2: value2

name of representation 2:
  type: some other type
  name: some other method
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
  npy/
    1.npz, ..., N.npz
  png/
    1.png, ..., N.png
```

## 5. Bonus

### 5.1: Exporting video from PNGs

After exporting pngs, use this command (requires `ffmpeg`)

```bash
old_path=`pwd`
cd /path/to/output_dir/collage
ffmpeg -start_number 1 -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p $oldPath/collage.mp4;
cd -;
```

### 5.2 Run in docker
- use `meehai/vre:latest` from docker hub.

```
mkdir example
# move the cfg and the video in some local dir
gdown https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk -O example/vid.mp4
wget https://gitlab.com/meehai/video-representations-extractor/-/raw/df15af177edf5c101bbb241428c43faac333cea4/test/end_to_end/imgur/cfg.yaml -O example/cfg.yaml
docker run \
  -v `pwd`/example:/app/resources \
  meehai/vre \
  /app/resources/vid.mp4 --cfg_path /app/resources/cfg.yaml -o /app/resources/result --start_frame 5 --end_frame 6
```

### 5.3 Batched experiment

Last updated at: 2023.11.11

See [implementation](examples/batched_experiment.py) for cfg and code.

|                        |    batch=1 |    batch=3 |   ratio 1/3 |    batch=5 |   ratio 1/5 |
|:-----------------------|-----------:|-----------:|------------:|-----------:|------------:|
| rgb                    | 0.00287311 | 0.00285926 |    1.00485  | 0.00355646 |    0.807857 |
| hsv                    | 0.227086   | 0.23222    |    0.977893 | 0.234075   |    0.970141 |
| normals svd (dpth)     | 3.05688    | 3.04678    |    1.00331  | 3.096      |    0.987362 |
| halftone               | 3.1492     | 3.14151    |    1.00245  | 3.13804    |    1.00356  |
| softseg kmeans         | 0.821758   | 0.818581   |    1.00388  | 0.816683   |    1.00621  |
| canny                  | 0.0149038  | 0.0143591  |    1.03793  | 0.0142797  |    1.0437   |
| softseg gb             | 0.256591   | 0.244486   |    1.04952  | 0.236025   |    1.08714  |
| dexined                | 0.121858   | 0.10822    |    1.12602  | 0.105016   |    1.16038  |
| depth dpt              | 0.134931   | 0.121348   |    1.11193  | 0.113324   |    1.19066  |
| depth odoflow (raft)   | 1.39505    | 1.17315    |    1.18915  | 1.15518    |    1.20764  |
| fastsam (x)            | 0.0565283  | 0.0414641  |    1.36331  | 0.0392864  |    1.43888  |
| opticalflow rife       | 0.0435189  | 0.0308007  |    1.41292  | 0.0288509  |    1.50841  |
| opticalflow raft       | 0.999548   | 0.742625   |    1.34597  | 0.650921   |    1.53559  |
| fastsam (s)            | 0.0348993  | 0.0215417  |    1.62008  | 0.019354   |    1.80321  |
| semantic safeuav torch | 0.0251322  | 0.012268   |    2.04859  | 0.00914755 |    2.74742  |

We can easily observe that some representations are not batched yet (all with ratios close to 1). We can also observe
that some of them are super slow (svd, halftone, odoflow).