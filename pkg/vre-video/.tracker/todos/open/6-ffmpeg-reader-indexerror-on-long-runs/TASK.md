# FFmpegFrameReader crashes with IndexError on long sequential runs

**Created**: 2026-08-05
**Priority**: 1

## Issue

Sequential reading of a video crashes near the end with `IndexError: list index out of range` in `FFmpegFrameReader.get_one_frame` (ffmpeg_frame_reader.py:129-133). Seen in VRE with `sift bs=1` on a 9022-frame video: crashed at ix=8934 (99%) after 48 minutes. Crash state: `cache_start_frame == cache_end_frame == ix`, empty cache, and the fallback retry at line 132 fails identically — the reader can never recover.

## Reproduction

100% reproducible, no special video needed:

```
ffmpeg -f lavfi -i testsrc=size=320x240:rate=30 -t 34 -c:v libx264 /tmp/repro.mp4
python3 -c 'import sys; sys.path.insert(0, "."); from vre_video.readers.ffmpeg_frame_reader import FFmpegFrameReader; r = FFmpegFrameReader("/tmp/repro.mp4"); [r[i] for i in range(len(r))]'
```

Crashes at `n - n/102` frames (ix=1010 of 1020) — same signature as the user report (8934 = 9022 - 88).

## Analysis (verified against the repro)

1. **Off-by-one per cache window** — the read loop in `_cache_frames` checks `len(self.cache) > self.cache_max_len` *after* consuming a frame from the pipe, so each window consumes `cache_max_len + 2` frames but accounts only `cache_max_len + 1` (`cache_end_frame = start_frame + len(cache)`). Bookkeeping drifts 1 frame behind the real stream per window; one real frame per window is also silently dropped.
2. **Dead-process continuation trap** — `_cache_frames` reuses the existing ffmpeg process when `start_frame == cache_end_frame` (continuation). Once ffmpeg reaches EOF, the drift from (1) makes a request land at `ix == cache_end_frame` with `ix < len(self)`, the read returns `b""` from the exhausted pipe → empty cache → IndexError, and the retry path takes the same branch forever.

## Notes

- Frames delivered before the crash are off by one per cache window (silent frame skip) — worth fixing too.
- Repro artifacts: /tmp/opencode/repro.py, repro3.py (window/process tracing).
