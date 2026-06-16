# Weights Repository: add weights_file for LearnedRepresentationMixin

**Created**: 2024-10-27
**Closed**: 2025-01-24
**Priority**: 3
**Labels**: refactor, weights repository

## Description

So we can get rid of the utils/weights_repository hardcoded stuff and move it into each class.

```
    # TODO: make this and get rid of the hardcoded stuff in utils.
    # @property
    # @abstractmethod
    # def weights_files(self) -> list[str]:
    #     """
    #     A list of files that must be present on the disk or downloaded from the weights repository. Only the stem is
    #     needed and vre_setup() must handle their loading. The data is stored under {weights_dir}/{repr_type}/[names]
    #     For example: 'depth/dpt' has 'depth_dpt_midas.pth' and is stored at 'weights_dir/depth/dpt/depth_dpt_midas.pth
    #     """

```
