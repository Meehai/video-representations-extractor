# Video Representation Extractor

![logo](logo.png)

## 1. Description
The purpose of this repository is to export the standard NGC representations from RGB alone.

<u>Supported representations</u>
  - Low level vision
    - RGB
    - HSV
    - Halftone
       - [python-halftone](https://github.com/philgyford/python-halftone)
    - Edges
      - [DexiNed](https://github.com/xavysp/DexiNed/)
  - Mid level vision
      - Unsupervised Depth
        - [Jiaw](TODO)
    - n/a
  - High level vision
    - n/a

<u>WIP representations</u>
  - Low level vision
    - TODO: representations & methods
  - Mid level vision
    - Unsupervised Depth
      - [Sfm learner](https://github.com/ClementPinard/SfmLearner-Pytorch)
      - TODO: more methods
    - Optical Flow
      - TODO: methods
  - High level vision
    - Semantic Segmentation
      - TODO: methods

## 2. Usage

`python main.py --videoPath /path/to/mp4 --cfgPath /path/to/cfg.yaml --outputDir /path/to/outputDir [--N n]`

The optional parameter N can be used in order to compute the first N frames of the video for debugging purposes.

## 3. CFG file
The config file will have the hyperparameters required to instantiate each supported method as well as global hyperparameters for the output. This means that if a depth method is pre-traied for 0-300m, this information will be encoded in the CFG file. Similarily, if the output resolution is 256x256, this will be encoded as a global hyperparameter.

Example cfg file:
```
resolution: 256,256
  rgb:
    method: rgb
    dependenceies: []
    parameters: None

  edges1:
    method: dexined
    dependencies: []
    parameters: None

  depth1:
    method: jiaw
    dependencies: [edges1]
    parameters:
      weightsFile: weights/dispnet_checkpoint.pth.tar
      resNetLayers: 18
      trainHeight: 256
      trainWidth: 448
      minDepth: 0
      maxDepth: 2
```

The above file will output 3 representations, named `rgb`, `edges1` and `depth1`, the second one based on `dexined` package with the specified parameters, while the third one based on `Jiaw` unsupervised depth pre-trained neural network, given the path to the model. The third one has as a dependency the second one, thus the second one will be computed in advance at every step.

Note: If the topological sort fails (because cycle dependencies), an error will be thrown.

## 4. Output format
All the outputs are going to be stored as [0-1] float32 npz files, one for each frame in a directory specified by `--outputDir`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:
```
/path/to/outputDir/
  rgb/
    1.npz, ..., N.npz + cfg.yaml
  edges1/
    1.npz, ..., N.npz + cfg.yaml
  depth1/
    1.npz, ..., N.npz + cfg.yaml
```

The `cfg.yaml` file for each representation is created so that we know what parameters were used for that representation.

## 5. Export PNGs
We can also export images as a grid collage of all exported representations by adding the CMD argument `--exportCollage=1` to the `main.py` script.

This yields a new directory with PNGs:
```
/path/to/outputDir/
  collage/
    1.png, ..., N.png
```

**Bonus**: Exporting video from PNGs
```
oldPath=`pwd`; cd /path/to/outputDir/collage; ffmpeg -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p $oldPath/collage.mp4; cd -;
```