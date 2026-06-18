# Merge IORepresentationMixin into Representation (IO by default)

**Created**: 2026-06-18
**Priority**: 3

## Problem

To implement a new output representation today you must inherit **both**
`Representation` AND `IORepresentationMixin` (in practice `NpIORepresentation`):

```python
class SemanticRepresentation(Representation, NpIORepresentation, ResizableRepresentationMixin): ...
class DepthRepresentation(Representation, NpIORepresentation, NormedRepresentationMixin, ...): ...
```

If you forget the IO mixin, the class still works as a `Representation` but is
**silently** not treated as an output representation. The whole VRE pipeline
gates on `isinstance(r, IORepresentationMixin)`:

- `video_representations_extractor.py:47` — set_io_params only on IO reprs
- `video_representations_extractor.py:154,223` — asserts IO before storing
- `representations_list.py:35` — filters output reprs by IO
- `build_representations.py:72,79,100` — IO param wiring
- `data_writer.py:12` — `Repr = Representation | IORepresentationMixin`

So missing the mixin = no error, just nothing gets written. Bad ergonomics,
easy footgun.

## Goal

A `Representation` should be an **output representation by default**. Merge IO
into the base so a new repr needs one base class, not a pile of mixins.

## Design questions (decide before implementing)

1. **Merge or inherit?** Either fold `IORepresentationMixin` into
   `Representation`, or have `Representation(IORepresentationMixin, ...)`.
   The IO methods (`load_from_disk`, `disk_to_memory_fmt`, `memory_to_disk_fmt`,
   `save_to_disk`) are `@abstractmethod` — making them abstract on the base
   forces every repr to implement them. `NpIORepresentation` already provides
   default npz/npy impls, so the realistic default is probably
   "Representation = current Representation + NpIORepresentation behavior".

2. **What about the other mixins?** `NormedRepresentationMixin`,
   `ResizableRepresentationMixin` are genuinely optional per-repr. Only IO is
   claimed "always needed". Keep those as mixins. The user's words:
   "Representation should already inherit IO and **something else**" — figure
   out what "something else" is (likely just IO/NpIO; confirm).

3. **Non-output representations?** Are there any reprs that are intentionally
   NOT outputs (pure intermediate/dependency)? If yes, IO-by-default needs an
   opt-out. If no, the `isinstance(IORepresentationMixin)` checks become
   redundant and can be removed/simplified.

## Affected files

- `vre/representations/representation.py` (base)
- `vre/representations/mixins/io_representation_mixin.py`
- `vre/representations/mixins/np_io_representation.py`
- `vre/representations/task_mapper.py`
- all of `vre_repository/*/...representation.py` (drop the explicit IO base)
- the `isinstance(..., IORepresentationMixin)` call sites above

## Acceptance

- New repr = inherit `Representation` only (+ optional Normed/Resizable) and it
  is an output representation automatically.
- Forgetting nothing silently breaks output writing.
- `bash test/e2e/run_all.sh` passes (parity).

## Notes

- Tester (Claude) does NOT edit non-test Python. Developer implements; once it
  lands, update affected unit tests / e2e expectations.
