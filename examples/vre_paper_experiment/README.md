# Paper experiment. Running VRE step by step from scratch on a new video.

### 1. Get a video

We'll use [this video](https://huggingface.co/datasets/Meehai/dronescapes2/resolve/main/raw_data/videos/new_videos/politehnica_DJI_0741_a2_540p.mp4) from the Dronescapes2 dataset.

### 2. Running VRE on RGB+HSV only

The basic config file:
```yaml
default_compute_parameters:
  batch_size: ${oc.decode:${oc.env:BATCH_SIZE}}

default_io_parameters:
  binary_format: npz
  image_format: ${oc.env:IMAGE_FORMAT}
  compress: ${oc.decode:${oc.env:COMPRESS}}
  output_size: ${oc.decode:${oc.env:OUTPUT_SIZE}}

default_learned_parameters:
  device: ${oc.env:VRE_DEVICE,cpu}

representations:
  rgb:
    type: color/rgb
    dependencies: []
    parameters: {}
  hsv:
    type: color/hsv
    dependencies: [rgb]
    parameters: {}
```

```bash
COMPRESS=False IMAGE_FORMAT=not-set OUTPUT_SIZE='[240,320]' BATCH_SIZE=10 vre video_240_320.mp4 -o data_240_320/ --config_path cfg_rgb_hsv.yaml --frames 0..100 --output_dir_exists_mode skip_computed
```

We run the following command:
```python
# main_args.py
image_format=["not-set", "jpg"]
resolutions=[[240,320], [540,960], [720,1280], [1080,1920]]
batch_sizes=[1,5,20]
num_frames=[100, 200, 300, 400, 500]
# args = [(i, os, 1, nf) for i in image_format for os in resolutions for nf in num_frames]

args = []
# I, OS, BS, NF
args.extend([(False, "not-set", resolution, 1, 100) for resolution in resolutions])
args.extend([(False, "jpg", resolution, 1, 100) for resolution in resolutions])
args.extend([(True, "jpg", resolution, 1, 100) for resolution in resolutions])

for c,i,os,bs,nf in args:
  print(f"COMPRESS={c} IMAGE_FORMAT={i} OUTPUT_SIZE='{os}' BATCH_SIZE={bs} vre video_{os[0]}_{os[1]}.mp4 -o data_c{c}_n{nf}_{bs}_{i}_{os[0]}_{os[1]}/ --config_path cfg_rgb_hsv.yaml --frames 0..{nf} --output_dir_exists_mode skip_computed")
```

### 3. Batch size experiment with RGB+DPT and RGB+SafeUAV

two configs:
- `cfg_rgb_safeuav.yaml`
- `cfg_rgb_dpt.yaml`

run this:
```bash
VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 vre video_540_960.mp4 -o safeuav_video_540_960 --config_path cfg_rgb_safeuav.yaml --output_dir_exists_mode skip_computed --exception_mode skip_representation --n_threads_data_storer 4 -I semantic_mapper.py:get_new_semantic_mapped_tasks --frames 0..100
```

### 4. Parallel with Dronescapes2 cfg

vre regular
```bash
VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 vre video_540_960.mp4 -o data_video_540_960 --config_path cfg_dronescapes2.yaml --output_dir_exists_mode skip_computed --exception_mode skip_representation --n_threads_data_storer 4 -I semantic_mapper.py:get_new_semantic_mapped_tasks --frames 0..100
```

vre parallel:
```bash
VRE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vre_gpu_parallel video_540_960.mp4 -o data_video_540_960_parallel --config_path cfg_dronescapes2.yaml --output_dir_exists_mode skip_computed --exception_mode skip_representation --n_threads_data_storer 4 -I semantic_mapper.py:get_new_semantic_mapped_tasks --frames 0..100
```

### 4. Streaming experiment

```bash

ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming - cfg_rgb.yaml --output_size 540 960 --input_size 540 960 --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_safeuav_150k.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_safeuav_430k.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_safeuav_1M.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_safeuav_4M.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_dpt.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_mask2former.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
ffmpeg -i video_540_960.mp4 -f rawvideo -pix_fmt rgb24 - | VRE_DEVICE=cuda vre_streaming -  cfg_rgb_marigold.yaml --output_size 540 1920 --input_size 540 960  --disable_hud --disable_async_worker > /dev/null
```