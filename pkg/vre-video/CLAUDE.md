# CLAUDE.md — VRE Video

Python video reader backed by ffmpeg, with PIL/numpy/raw-fd backends too. Frames are numpy arrays; indexing `video[ix]` works like a list. A tkinter video player is shipped as a script. Used by VRE (modality extraction) and drone-ioact.

**Keep answers clean, short. Check the code before answering — never answer from memory. Read the file you're asked about; it may have changed.**

## Layout

```
vre_video/                 # the package
├── vre_video.py           # VREVideo: the only public class (readers/writers behind it)
├── utils.py               # logger, image_read/image_write, image_add_text (PIL helpers)
├── readers/               # FrameReader base + 4 impls + build_frame_reader() dispatch
│   ├── frame_reader.py    #   FrameReader ABC (shape/fps/path, __getitem__, __len__)
│   ├── ffmpeg_frame_reader.py  #   FFmpegFrameReader + FFprobe (mp4/mkv/URLs/v4l2 probe)
│   ├── pil_frame_reader.py     #   dir of png/jpg
│   ├── numpy_frame_reader.py   #   dir of npy/npz, in-memory ndarray, list of ndarrays
│   └── fd_frame_reader.py      #   FdFrameReader: raw frames from stdin/pipe/socket (async worker thread)
└── writers/                # FrameWriter base + 3 impls + build_frame_writer() dispatch
    ├── frame_writer.py
    ├── ffmpeg_frame_writer.py
    ├── pil_frame_writer.py
    └── numpy_frame_writer.py
cli/vre_video_player.py    # tkinter player (installed via setup.py scripts=) — mirrors examples/vre-video-player/
examples/
├── vre-video-player/      # the same dummy video player, self-contained
└── video-live-coding/     # live-coding demo (main.py + src.py)
test/
├── unit/readers/          # per-reader tests + build_frame_reader dispatch tests
├── unit/writers/
└── integration/           # end-to-end VREVideo read/write tests
.tracker/todos/            # task tracker, mirrors GitLab work items (see below)
setup.py                   # packaging; scripts=[cli/vre_video_player.py]
requirements.txt           # dev deps (drifts from setup.py on purpose)
.gitlab-ci.yml             # CI: pylint (excludes test/, examples/) + pytest
```

## How it works

- `VREVideo(source)` → `build_frame_reader(source)` auto-detect: FrameReader → itself; `np.ndarray`/list → `NumpyFrameReader`; `IOBase` → `FdFrameReader`; `"-"` → `FdFrameReader(sys.stdin.buffer)` (stdin like ffplay); `/dev/*` → v4l2 via ffmpeg pipe (`_build_dev_video_ffmpeg` in `readers/__init__.py`); directory → PIL (png/jpg) or numpy (npy/npz); anything else → `FFmpegFrameReader` (ffmpeg errors surface as-is, YouTube URLs via youtube-dl).
- `VREVideo.write()` picks the writer: suffix on the output path → ffmpeg, else `build_frame_writer()` which defaults to the `VRE_VIDEO_DEFAULT_WRITER` env var ("pillow").
- **Runtime dep outside pip: `ffmpeg` in PATH.** pip deps: numpy, Pillow, loggez, tqdm, overrides (pinned).
- Frame readers return `HxWx3` uint8 arrays; `FdFrameReader` reads raw bytes in an async thread (set `async_worker=False` to read everything — at your own memory risk).

## Testing

```bash
python -m pytest test          # unit + integration (same as CI)
python -m pylint --rcfile=.pylintrc $(git ls-files "*.py" | grep -v "test/\|examples/")
```

## Task tracker

Tasks live in `.tracker/todos/` with `open/` and `closed/` subdirs — a mirror of the GitLab work-item board. One task = one dir `NN-slug/` containing `TASK.md` (`NN` is the upstream work-item iid, global across open/closed). Status comes from the directory, not the file. Header format:

```
# Task title

**Created**: YYYY-MM-DD
**Closed**: YYYY-MM-DD   (closed tasks only)
**Priority**: 1
```

Then a short body with whatever context matters, plus the upstream work-item link. When a task is done, move its dir from `open/` to `closed/` and add the `**Closed**` field. Keep the mirror in sync with upstream (e.g. when an upstream issue closes, close it here).
