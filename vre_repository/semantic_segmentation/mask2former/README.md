# Mask2Former in VRE

## 1. Convert official weights to the regular torch format

Go to [link](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).

Download the config file (yaml) and the .pickle weights file.

Call the converter script:
```bash
python convert.py config.yaml weights.pickle weights.ckpt
```

## 2. Run the semantic segmentation

```bash
python mask2former.py /path/to/weights.ckpt input.jpg output.jpg
```

We already suport ootb 3 weights: `47429163_0`, `49189528_1` and `49189528_0`. You can call the script as:
```bash
python mask2former.py 47429163_0 input.jpg output.jpg
```

and it should work ootb.
