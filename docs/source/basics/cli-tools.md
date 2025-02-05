# CLI tools

## vre

TODO

## vre_collage

`vre_collage` that takes all the image files (png, jpg etc.) from an output_dir as above and
stacks them together in a single image. This is useful if we want to create a single image of all representations which
can later be turned into a video as well.

Usage:
```
vre_collage /path/to/output_dir --config_path /path/to/config.yaml -o /path/to/collage_dir
[--overwrite] [--video] [--fps] [--output_resolution H W]
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
```
vre_reader /path/to/output_dir --config_path /path/to/config.yaml --mode [MODE] [--batch_size B]
[--handle_missing_data H]
```

We have 2 choices for `--mode`:
- `read_one_batch` Reads a single batch at random using `torch.utils.data.DataLoader` and, after reading, it will invoke
the plotting function (similar to `vre_collage`). It only prints the shapes.
- `iterate_all_data` Will iterate the entire dataset to see if the data is corrupted.

Handle missing data is passed to the `MultiTaskDataset` dataset constructor and accepts all the choices from there.
Defaults to 'raise' which will raise an exception if any representation has missing data (i.e. each xxx.npz must be
present for all the representations).
