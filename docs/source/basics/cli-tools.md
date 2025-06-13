# CLI tools

## vre

The main CLI tool used. Usage:

```bash
vre
  VIDEO
  -o /path/to/output_dir # or --output_path
  --config_path /path/to/config.yaml
  [--representations r1 r2 ]
  [--frames F1 F2 F5] or [--frames F1..F10]
  [--output_dir_exists_mode {overwrite,skip_computed,raise}]
  [--exception_mode {skip_representation,stop_execution}]
  [--n_threads_data_storer N]
  [-I f1.py:foo f2.py:foo2] # or --external_representations
  [-J f1.py:foo f2.py:foo2] # or --external_repositories
  [--collage] # calls vre_collage CLI tool at the end with the args below
    [--collage_videp] # if --collage is set
      [--collage_fps] # required if -collage is set and --collage_video is set
```

Lots of optional arguments for sure, but we'll see examples later on.

## vre_collage

`vre_collage` that takes all the image files (png, jpg etc.) from an output_dir as above and
stacks them together in a single image. This is useful if we want to create a single image of all representations which
can later be turned into a video as well.

Usage:
```bash
vre_collage
  /path/to/output_dir
  --config_path /path/to/config.yaml
  -o /path/to/collage_dir
  [--overwrite]
  [--video]
  [--fps]
  [--output_resolution H W]
```

```
/path/to/collage_dir/
  collage/
    1.png, ..., N.png # 1.png = stack([name_of_representation/png/1.png, name_of_representation_2/png/1.png])
```

Note: you can also get video from a collage dir like this (in case you forgot to set --video or want more control):

```bash
cd /path/to/collage_dir
ffmpeg -start_number 1 -framerate 30 -i %d.png -c:v libx264 -pix_fmt yuv420p /path/to/collage.mp4;
```

## vre_reader

The `vre_reader` tool can be used to iterate over the extracted vre dataset from the provided output path.

Usage:
```bash
vre_reader
  /path/to/output_dir
  --config_path /path/to/config.yaml
  --mode [MODE]
  [--batch_size B]
  [--handle_missing_data H]
```

We have 2 choices for `--mode`:
- `read_one_batch` Reads a single batch at random using `torch.utils.data.DataLoader` and, after reading, it will invoke the plotting function (similar to `vre_collage`). It only prints the shapes.
- `iterate_all_data` Will iterate the entire dataset to see if the data is corrupted.

Handle missing data is passed to the `MultiTaskDataset` dataset constructor and accepts all the choices from there.
Defaults to 'raise' which will raise an exception if any representation has missing data (i.e. each xxx.npz must be
present for all the representations).

## vre_dir_analysis

Generates a json file (printed to stdout) with the status of a VRE base dir. A VRE base dir contains one or more vre output dirs of 1 or more videos.

So, for `base/vre_video_1` and `base/vre_video_2` you would get a dashboard of 2 rows and N representataons (outer join) of these two videos. Useful when working with more videos in parallel to keep a global track

Usage:

```bash
vre_dir_analysis root_dir > res.json`
```

`root_dir` must be like: `[subdir1/[repr1,...,reprn], subdir2[], ... ]`

## vre_gpu_parallel

A script that tries to shard a single video across multiple gpus by dividing the frames equally as per the `--gpus` argument. The outputs are written to the same final VRE directory, so all the other arguments are sent to VRE. If `--frames` is sent to this script, then these frames are used to shard, otherwise the entire video.

Usage:

```bash
CUDA_VISIBLE_DEVICES 0,1,2,... vre_gpu_parallel [--frames F1..FN]  <all the other vre args>
```

## vre_streaming

A script that reads frame by frame (or by batches) from a video and a vre config and outputs to various external tools used for streaming: matplotlib, ffplay, mpv/vlc (via ffmpeg+tcp) or html5 (via ffmpeg+HLS). By default it outputs the image raw bytes to stdout, so be careful :)

Usage:
```bash
MPL=1 vre_streaming test_video.mp4 # matplotlib, requires no external tools
vre_streaming VIDEO.mp4 | ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x360 -framerate 30 -i - # ffplay
```

See [here](examples/vre_streaming/README.md) for more examples.
