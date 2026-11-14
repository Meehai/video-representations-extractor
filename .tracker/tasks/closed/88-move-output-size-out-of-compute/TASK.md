# Move output_size/output_dtype out of ComputeRepresentation (and into Representation or IORepresentation?)

**Created**: 2024-12-25
**Priority**: 3
**Closed**: 2025-01-11

## Description

We import ComputeRepresentation and override compute() with a dummy method in a lot of places and it's clearly the wrong pattern.
