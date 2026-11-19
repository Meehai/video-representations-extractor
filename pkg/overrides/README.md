# overrides (first-party shim)

> **FIRST-PARTY CODE — not the third-party `overrides` package.**
> A small in-house reimplementation of the `@overrides` / `@override` decorator providing only the
> **existence check** robosim relies on: at class-definition time it asserts that the decorated
> method overrides a same-named method in a base class, raising `TypeError` otherwise. It lives
> under `pkg/` so `from overrides import overrides` resolves to it with no call-site changes, but it
> is maintained by us — there is no upstream and no LICENSE to carry.

## What it does NOT do

The upstream package (7.7.0) checks **signature compatibility** via a ~780-line typing engine
(`typing_utils` + `signature`) and ships `EnforceOverrides` / `@final` — none of that, here. Instead
of a variance/subtyping analysis, this shim compares the override's signature to the base's *as
written* (see the module docstring for how annotations are keyed). The opt-out keyword is the same
one upstream ships — `check_signature` — so source written against this shim runs identically
against a pip-installed `overrides` in the conda env, and vice versa. `check_at_runtime` is not
implemented (this shim never checks at runtime).

## Usage

```python
from overrides import overrides

class Base:
    def run(self) -> None: ...
    def step(self, dt: float) -> float: ...

class Impl(Base):
    @overrides
    def run(self) -> None: ...   # raises TypeError at class definition if Base has no `run`
                                 # or if the signature differs (name/kind/default/annotation/return)

    @overrides(check_signature=False)
    def step(self, dt: float, t: float = 0.0) -> float: ...   # opts out of the signature check
```

## History

Was a byte-faithful vendor of overrides 7.7.0 (Apache-2.0); trimmed to this first-party shim (task
167's "Later" note). The mechanism that finds base classes — disassembling the class-body frame to
recover base names — is copied from upstream; the signature engine is not. Task 183 briefly renamed
the opt-out flag to `changed_parameters`; that was reverted to upstream's `check_signature` (noted
in task 183) because the pip `overrides` package is also present in the conda env and the two must
agree on the keyword.
