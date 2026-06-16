# Potential Bug: External representations may not properly work when depending on VRE representations

**Created**: 2024-11-16
**Closed**: 2024-12-22
**Priority**: 1
**Labels**: bug

## Description

To be analyzed and understood (and maybe a test done) if I have:
- Repr1 (in VRE)
- Repr2 depending on Repr1 (outside)

How do these work? I think right now we have a duplicate Repr1 in external (because they need to be instantiated) and somehow the assert doesn't trigger as well (why?).
