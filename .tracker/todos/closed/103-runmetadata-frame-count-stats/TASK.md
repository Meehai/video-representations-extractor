# Enhancements to RunMetadata: add stats w.r.t the number of frames computed in that run

**Created**: 2025-03-10
**Closed**: 2025-03-11
**Priority**: 3

## Description

```json
{
    "id": "oKcZCcYa63",
    "runtime_args": {
        "video_path": "/home/mihai/code/ml/video-representations-extractor/test/vre_repository/end_to_end/imgur/test_video.mp4",
        "video_shape": "(5395, 720, 1280, 3)",
        "video_fps": "29.97",
        "representations": [
            "rgb",
            "halftone1",
            "edges_canny",
            "softseg_gb",
            "edges_dexined",
            "opticalflow_rife",
            "fastsam(s)",
            "mask2former",
            "depth_dpt",
            "safeuav",
            "hsv",
            "normals_svd(depth_dpt)"
        ],
        "frames": [
            ...
        ],
        "exception_mode": "stop_execution",
        "n_threads_data_storer": 0
    },
    "data_writers": {
        "rgb": {
    ...
    }, # TO BE ADDED
    "run_stats": {
      "rgb": { "n_computed": N, "n_skipped": M, "n_failed": K, "avg computed duration(?)": T },
      ....
    }
}
```
