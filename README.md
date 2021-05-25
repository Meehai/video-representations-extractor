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
representations:
  halftone:
    method: python-halftone
    name: halftone1
    parameters:
      sample: 3
      scale: 1
      percentage: 91
  depth:
    method: sfmlearner
    name: depth1
    parameters:
      weightsFile: /path/to/model.pkl
```

The above file will output 2 representations, named `halftone1` and `depth1`, the first one based on `python-halftone` package with the specified parameters, the second one based on SfmLearner pre-trained neural network, given the path to the model.

## 4. Output format
All the outputs are going to be stored as [0-1] float32 npz files, one for each frame in a directory specified by `--outputDir`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:
```
/path/to/outputDir/
  
  halftone1/
    1.npz, ..., N.npz + cfg.yaml
  
  depth1/
    1.npz, ..., N.npz + cfg.yaml
```

The `cfg.yaml` file for each representation is created so that we know what parameters were used for that representation.

## 5. Export PNGs
Given a exported video, as npz files, we can create a collage of stacked image representations, which is useful to present the outputs in a human viewable format.

Usage:
`python main_make_collage.py --videoPath /path/to/mp4 --cfgPath /path/to/cfg.yaml --outputDir /path/to/outputDir`

This yields a new directory with PNGs:
```
/path/to/outputDir/

  collage/
    1.png, ..., N.png
```
