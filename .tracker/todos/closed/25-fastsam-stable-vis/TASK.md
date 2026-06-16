# fastsam stable visualisation

**Created**: 2023-11-11
**Closed**: 2025-03-10
**Priority**: 2
**Labels**: docs/examples/user api, feature

## Description

- add a 3rd parameter set -- visualisation_params?
- few variants:
  - still use the existing ones (random and transparent)
  - black and white (black = background, white = foreground)
  - just foreground is segmented with a fixed color
  - stable color across time: for each bbox, use the median pixel for the color (use the same logic as in kmeans)
