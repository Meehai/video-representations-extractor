# VRE basic concepts

## High level

The purpose of this repository is to execute the following very simple pseudo-code reliably and hopefully efficient:

```python
videos = ["1.mp4", "2.mkv"]
representations = ["rgb", "hsv", "camera_normals(depth)", "depth", "semantic_segmentation"]
for video in videos: # this loop is handled by the user
  tsr = topo_sort(representations) # [{"level 1": [rgb, hsv, depth, semantic_segmentation], "level 2": [camera_normals(depth)]}
  for representation in tsr:
     for frame in frames(video):
       y_repr = representation(video, frame)
       store_to_disk(y_repr, output_dir / representation / frame)
```

- The `for video in videos` must be handled by the user, i.e. the VRE tool operates on a single video at a time.
- The `topo_sort(representations)` is defined via dependencies, as some representations (i.e. camera normals) may require others
to be pre-computed (depth) and available to be loaded from disk.
- The `for frame in frames(video)` is batched in VRE. Each representation can define it's own batch size for efficiency. Furthermore, the batches may be completely skipped if they are already pre-computed (i.e. present on the disk)
- The `store_to_disk()` part can also be configurable (npy, npz, npz+compression, png, etc.)

The tool is designed to be single processing at the moment, so in case you are lucky to have access to more than 1 GPU, you can optimize your load in two ways:
- if you have >1 video, use 1 video per GPU
- if you have more GPUs available, you can specify subsets of representations via the `--representations` flag manually pointing to the same `output_dir` as a subdir will be created for each reprsentation (see pseudo-code)
- otherwise, if you have a single reprsentation you want to compute and a single video, you can divide the frames yourself via `--frames` which accepts a list of frames or a `start..end` syntax.
- There is some work in progress to allow some automatic multi-processing scheduling (i.e. divide the video in 4 if you have 4 GPUs or divide the representations w.r.t the topo sort automatically or both). But for the moment, this part must be done by hand.

## Config file

The config file will have the hyperparameters required to instantiate each supported method as well as global
hyperparameters for the output. These parameters are sent to the constructor of each representation, so one can pass
additional semantics to each representation, such as classes of a semantic segmentation or the maximum global depth
value in meters.

High level format:

```yaml
name_of_representation: # spaces work too in the name but it's less desired due to other tools having issues
  type: some high level type (such as depth/dpt, semantic/mask2former, edges/dexined etc.)
  dependencies: [a list of dependencies given by their names]
  parameters: # as defined in the constructor of the implementation
    param1: value1
    param2: value2
  learned_parameters: # applies only to representations that are learned and require weights (only torch atm)
    device: "cuda" # for representations that have in their vre_setup() method a model.to(device) call

name_of_representation_2:
  type: some other type
  name: some other method
  dependencies: [name_of_representation] # since this representation depends on the prev one, it'll be computed after
  parameters: []
  compute_paramters:
    batch_size: 5 # overwrite the global default (that is 1)
  io_parameters:
    image_format: not-set
    binary_format: npy # overwrite the global default (that is npz)
    compress: False
```

Example cfg file: See [out of the box supported representations](test/end_to_end/imgur/cfg.yaml) and the CFG defined
in the [CI process](test/end_to_end/imgur/run.sh) for an actual export that is done at every commit on a real video.

Note: If the topological sort fails (because of cycle dependencies), an error will be thrown.

## Output format

All the outputs are going to be stored in the output directory (`--output_dir/-o`) with one file for each frame.
A subdirectory will be created for each representation. We have 2 options right now: binary_format (npz or npy only) and
image_format (png, jpg, whatever is supported by the PIL writer etc.) though we might merge this into a single
'exporter' thsat will receive a list of possible output formats for each representation.

For the above CFG file, 2 subdirectories will be created:

```
/path/to/output_dir/
  name_of_representation/
    npz/ # if binary_format=='npz'
      1.npz, ..., N.npz
    png/ # if image_format=='png'
      1.png, ..., N.png
  name_of_representation_2/
    npz/
      1.npz, ..., N.npz
```

The `cfg.yaml` file for each representation is created so that we know what parameters were used for that
representation.

## Relevant env variables

- `VRE_WEIGHTS_DIR` The path where the weights are downloaded from the weights repository. If not set, will be defaulted
in the root of the project in a resources dir.
- `VRE_DEVICE` Some default cfgs will use this to set the device of representations. Usage is:
 `VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 vre ...`. You can of course update the yaml config files to not use this
 pattern.
