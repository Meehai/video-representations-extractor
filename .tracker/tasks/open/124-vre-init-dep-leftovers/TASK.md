# vre/__init__.py: drop diffusers + transformers leftovers

**Created**: 2026-11-19
**Priority**: 3
**Closed**: n/a

**Labels**: deps

## Description

Core leftovers, kept out of the #123 MR (!255) on purpose (it only touches `vre_repository/depth/marigold/` + `setup.py`):

- `vre/__init__.py`: `try: from diffusers.utils.logging import disable_progress_bar` -- core must not know about
  diffusers at all. Delete it (and drop `wrong-import-position` from the pylint disable if nothing else needs it).
- `vre/__init__.py`: `warnings.filterwarnings("ignore", "`clean_up_tokenization_spaces` was not set*.")` is a
  transformers leftover. Delete.

!255 is merged: the diffusers import is dead code now.
