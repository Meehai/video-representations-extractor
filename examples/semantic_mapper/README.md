# Semantic Mapper

Experimental ground for new generic/low-level representations as combinations higher-level representations.

Up to date repo: [link](https://www.gitlab.com/video-representations-extractor/semantic-mapper).

For example:

```
- safe_landing(no_water) <- semantic_m2f_coco_converted <- [semantic_m2f_coco]
                         <- semantic_m2f_mapillary_converted <- [semantic_m2f_mapillary]
                         <- normals_svd_depth_marigold <- [depth_marigold]
                         <- water_sky <- semantic_m2f_coco_converted <- [semantic_m2f_coco]
                                      <- semantic_m2f_mapillary_converted <- [semantic_m2f_mapillary]
```

- `[xxx]` is denoted by an expert, while a regular one is derived from 1 or more experts
