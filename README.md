# Video Representation Extractor

![logo](logo.png)

## 1. Description
The purpose of this repository is to export the standard NGC representations from RGB alone.

<u>Supported representations</u>
 - See [here](https://gitlab.com/neural-graph-consensus/video-representations-extractor/-/blob/master/video_representations_extractor/representations/get_representation.py) for a comprehensive list, since it updates faster than this README.

Weights repository for pretrained networks is [here](https://drive.google.com/drive/folders/1bWKEAiTXDpgaY2YOAFBvMqqyOGSafoIm?usp=sharing).

## 2. Usage

`python main.py --videoPath /path/to/mp4 --cfgPath /path/to/cfg.yaml --outputDir /path/to/outputDir [--startFrame a] [--endFrame b]`

The optional parameter N can be used in order to compute just as pecific interval of the video, for testing and skipping irrelevant areas, if needed.

## 3. CFG file
The config file will have the hyperparameters required to instantiate each supported method as well as global hyperparameters for the output. This means that if a depth method is pre-traied for 0-300m, this information will be encoded in the CFG file. Similarily, if the output resolution is 256x256, this will be encoded as a global hyperparameter.

High level format:
```
name of representation:
  type: some high level type (such as depth, semantic, edges, etc.)
  method: the implemented method
  dependencies: [a list of dependencies given by their names]
  [saveResults: resized_only] # optional parameter with possible values all, resized_only and none
  parameters: (as defined in the constructor of the implementation)
    param1: value1
    param2: value2

name of representation 2:
  type: some other type
  method: some other method
  dependencies: [name of representation]
  parameters: []
```

Example cfg file: See [out of the box supported representations](cfgs/testCfg_ootb.yaml)

Note: If the topological sort fails (because cycle dependencies), an error will be thrown.

## 4. Output format
All the outputs are going to be stored as [0-1] float32 npz files, one for each frame in a directory specified by `--outputDir`. A subdirectory will be created for each representation.

For the above CFG file, 2 subdirectories will be created:
```
/path/to/outputDir/
  name of representation/
    1.npz, ..., N.npz + cfg.yaml
  name of representation 2/
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
oldPath=`pwd`; cd /path/to/outputDir/collage; ffmpeg -start_number 1 -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p $oldPath/collage.mp4; cd -;
```
