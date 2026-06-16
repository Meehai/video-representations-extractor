# design/refactor idea: VRE object without reference to self.video

**Created**: 2025-08-29
**Closed**: 2025-08-30
**Priority**: 3

## Description

might simplify VRE Streaming and make the vre object not dependent on self.video

In theory we could work only at frames level, or pass the video reference not from self, but somehow from the caller.
