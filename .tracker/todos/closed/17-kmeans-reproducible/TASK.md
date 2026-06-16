# change kmeans implementation from cv2 -- it's not reproductible

**Created**: 2023-11-01
**Closed**: 2024-10-06
**Priority**: 2
**Labels**: representation

## Description

linked to #16 

integration test (batched vs not batched) doesn't work as kmeans returns different clusters every time and throws errors if we provide initial centroids :smile:
