# Video Representation Extractor

![logo](logo.png)

Documentation at: [link](https://video-representations-extractor.gitlab.io/video-representations-extractor/).

### Installation

```python
conda create -n vre python=3.12 anaconda # CI runs 3.12; setup.py requires >=3.11
pip install video-representations-extractor
[VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0] vre example/video.mp4 --connfig_path example/cfg.yaml -o example/
```

For more details, see the [usage and installation page](docs/source/introduction-and-installation/PAGE.md).

### Google Colab examples

- Batched VRE: Full example from a single test video to complex extracted representations: [link](https://colab.research.google.com/drive/1vAp71H-TLewhF56odv33TkmGwwhuoFJ-?usp=sharing)
- Streaming VRE: Real time Semantic Segmentation on youtube videos: [link](https://colab.research.google.com/drive/1-Zc7qEC7k7KnCuq-LOnNy9hoMfQBJA9O#scrollTo=TR5G9bBhY9_k)
