# change kmeans implementation from cv2 -- it's not reproductible

**Created**: 2023-11-01
**Priority**: 2
**Closed**: 2024-10-06

**Labels**: representation

## Description

linked to #16 

integration test (batched vs not batched) doesn't work as kmeans returns different clusters every time and throws errors if we provide initial centroids :smile:
