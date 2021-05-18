# Video Representation Maker

## 1. Description
The purpose of this repository is to export the standard NGC representations from RGB alone.

The common representations are:
  - Low level vision
    - HSV
    - Semantic GB
    - Halftone (via python-halftone)
    - Edges
  - Mid level vision
    - Unsupervised Depth (Sfm Learner style)
    - Optical Flow
  - High level vision
    - Semantic Segmentation (pretrained)

TODO: define more representations.

## 2. Usage

`python main_video.py --videoPath /path/to/mp4 --cfgPath /path/to/cfg.yaml --outputDir /path/to/outputDir`

## 3. CFG file
The config file will have the hyperparameters required to instantiate each supported method as well as global hyperparameters for the output. This means that if a depth method is pre-traied for 0-300m, this information will be encoded in the CFG file. Similarily, if the output resolution is 256x256, this will be encoded as a global hyperparameter.

Example cfg file:
```
resolution: 256,256
representations: [
  halftone:
    method: python-halftone
    name: halftone1
    parameters: {
      sample: 3,
      scale: 1,
      percentage: 91
    },

  depth:
    method: sfmlearner
    name: depth1
    parameters: {
      weightsFile: /path/to/model.pkl
    }
]
```

The above file will output 2 representations, named `halftone1` and `depth1`, the first one based on `python-halftone` package with the specified parameters, the second one based on SfmLearner pre-trained neural network, given the path to the model.

## 4. Output format
All the outputs are going to be stored as [0-1] float32 npy files, one for each frame in a directory specified by `--outputDir`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:
```
/path/to/outputDir/
  
  halftone1/
    1.npy, ..., N.npy
  
  depth1/
    1.npy, ..., N.npy
```