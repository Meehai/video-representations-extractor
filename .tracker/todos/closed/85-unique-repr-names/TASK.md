# Representations should be unique given the same name (even in dependencies)

**Created**: 2024-12-18
**Closed**: 2024-12-22
**Priority**: 1
**Labels**: bug

## Description

We have this weird issue with external dependencies
```bash
vre DJI_0708.MP4 --config_path /export/home/proiecte/aux/mihai_cristian.pirvu/datasets/dronescapes-2024/vre_dronescapes/cfg.yaml -o /tmp/tmpfq3y62ta --frames 0 --representations semantic_mask2former_coco_47429163_0 semantic_mask2former_mapillary_49189528_0 semantic_mask2former_mapillary_49189528_1 depth_marigold buildings "buildings(nearby)" containing rgb safe-landing-no-sseg safe-landing-semantics sky-and-water transportation vegetation "normals_svd(depth_marigold)" -I /export/home/proiecte/aux/mihai_cristian.pirvu/datasets/dronescapes-2024/scripts/semantic_mapper/semantic_mapper.py:get_new_semantic_mapped_tasks --output_dir_exists_mode skip_computed
```


```
(Pdb++) pprint(representations)
{'buildings': BinaryMapper(buildings ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1']),
 'buildings(nearby)': BuildingsFromM2FDepth(buildings(nearby) ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1', 'depth_marigold']),
 'containing': BinaryMapper(containing ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1']),
 'depth_marigold': Marigold(depth_marigold),
 'normals_svd(depth_marigold)': DepthNormalsSVD(normals_svd(depth_marigold) ['depth_marigold']),
 'rgb': RGB(rgb),
 'safe-landing-no-sseg': SafeLandingAreas(safe-landing-no-sseg ['depth_marigold', 'normals_svd(depth_marigold)']),
 'safe-landing-semantics': SafeLandingAreas(safe-landing-semantics ['depth_marigold', 'normals_svd(depth_marigold)', 'semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1']),
 'semantic_mask2former_coco_47429163_0': Mask2Former(semantic_mask2former_coco_47429163_0),
 'semantic_mask2former_mapillary_49189528_0': Mask2Former(semantic_mask2former_mapillary_49189528_0),
 'semantic_mask2former_mapillary_49189528_1': Mask2Former(semantic_mask2former_mapillary_49189528_1),
 'sky-and-water': BinaryMapper(sky-and-water ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1']),
 'transportation': BinaryMapper(transportation ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1']),
 'vegetation': BinaryMapper(vegetation ['semantic_mask2former_mapillary_49189528_0', 'semantic_mask2former_coco_47429163_0', 'semantic_mask2former_mapillary_49189528_1'])}

(Pdb++) type(representations["normals_svd(depth_marigold)"])
<class 'vre.representations.normals.depth_svd.depth_normals_svd.DepthNormalsSVD'>

(Pdb++) representations["safe-landing-no-sseg"].dependencies
[DepthRepresentation(depth_marigold), NormalsRepresentation(normals_svd(depth_marigold))]

(Pdb++) type(representations["safe-landing-no-sseg"].dependencies[1])
<class 'vre.representations.cv_representations.NormalsRepresentation'>

```

This should first and foremost fail (most likely in add external representations code) and then we think about how to solve it properly.
