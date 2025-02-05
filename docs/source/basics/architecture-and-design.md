# VRE basic concepts

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
