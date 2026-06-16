# Dynamically add ComputeRepresentation or StoredRepresentation at run() time based on disk information

**Created**: 2024-10-27
**Closed**: 2024-11-09
**Priority**: 3

## Description

No need for all the reprs to extend `ComputeRepresentation` and thus require batch_size/output_size/binary_format/image_format/output_dtype and all that if the data is stored on the disk
 - maybe it's good to have them to ensure loaded/stored is the same (or compatible, i.e. resized is upscaled so it can be downscaled again or smth)
