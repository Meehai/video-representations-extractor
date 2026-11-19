"""Unit tests for the first-party `overrides` shim (existence check + exact-signature check).

Covers the common cases robosim actually uses (server/server.py, server/plugins/*,
client/robosim_env.py — 32 sites, all bare `@overrides`): a method that overrides a
base method, the `@property`/`@overrides` and `@staticmethod`/`@overrides` stacks,
docstring inheritance, and the `TypeError` raised when nothing is overridden. Signature
check: an override must match the base's params + annotations + return type EXACTLY (no
contravariance/variance analysis — just "same as written"); a genuinely different
signature needs `@overrides(check_signature=False)` — the same keyword the third-party
`overrides` package uses — which skips the check entirely and tags the method. They also
pin the trim — only `overrides`/`override` are exported, no `EnforceOverrides`/`@final`.

The shim recovers base-class NAMES by disassembling the class statement's frame, so the
bases it can see must be module globals (LOAD_GLOBAL/LOAD_NAME) — exactly as in the real
code, where bases are imported names. Local base classes (LOAD_FAST) are invisible to it
by design, so all bases here live at module level.

Loaded hermetically by file path so we test THE VENDORED shim, not a pip-installed
`overrides` that may shadow it on sys.path (the local dev case).
"""
from __future__ import annotations

import collections
import collections.abc
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_INIT = Path(__file__).resolve().parents[1] / "overrides" / "__init__.py"


def _load_vendored():
    name = "robosim_vendored_overrides"
    spec = importlib.util.spec_from_file_location(
        name, _INIT, submodule_search_locations=[str(_INIT.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec so the relative import resolves
    spec.loader.exec_module(module)
    return module


_ov = _load_vendored()
_ov_inner = importlib.import_module(f"{_ov.__name__}.overrides")
overrides = _ov.overrides
override = _ov.override


# --- module-level bases (LOAD_GLOBAL — what the frame walker can resolve) --------------
class Base:
    def run(self) -> str:
        """base docstring."""
        return "base"

    def to_dict(self) -> dict:
        return {}


class Other:
    def extra(self) -> str:
        return "other"


# Dotted-type future-import asymmetry: this module HAS `from __future__ import annotations`, so the
# base's `frame: np.ndarray` is stored as the string "np.ndarray". A non-future module stores the
# live numpy.ndarray type. Same type, two spellings — the override in the regression test is exec'd
# in a fresh non-future namespace to mirror the real robochan split (screen_displayer_utils.py has
# the future import, the tkinter/sdl2 backends don't).
class _DottedBackend:
    def update_frame(self, frame: np.ndarray):
        pass


# Variadic tuple: future-imported here, so `-> tuple[int, ...]` is stored as the string
# "tuple[int, ...]"; a non-future override stores the live alias with an Ellipsis arg.
class _VariadicBase:
    def size(self) -> tuple[int, ...]:
        return ()


class GrandParent:
    def deep(self) -> str:
        return "grandparent"


class Parent(GrandParent):
    pass


# Base -> Derived1(check_signature=False) -> Derived2: proves Derived2 checks against its
# immediate base (Derived1), not the root. Module-level so the frame walker can see them.
class _Root:
    def run(self) -> str:
        return "root"


class _DerivedChanged(_Root):
    @overrides(check_signature=False)
    def run(self, x: int) -> str:
        return f"d1:{x}"


class Derived2(_DerivedChanged):
    @overrides
    def run(self, x: int) -> str:
        return f"d2:{x}"


# Covariant-return bases (module-level so the frame walker can resolve them), mirroring the real
# from_dict -> narrowed-state pattern in plugins/race_plugin.py and robosim/simulator_object.py.
class CovSerializable:
    @staticmethod
    def from_dict(state: dict) -> CovSerializable:
        return None


class CovRaceState(CovSerializable):
    @staticmethod
    @overrides
    def from_dict(state: dict) -> CovRaceState:
        return None


class CovSimSerializable:
    @staticmethod
    def from_dict(state: dict) -> CovSimSerializable:
        return None


class CovSimState(CovSimSerializable):
    @staticmethod
    @overrides
    def from_dict(state: dict) -> CovSimState:
        return None


class CovCodec:
    def encode(self, resp: dict) -> bytes:
        return b""


# Mirror of server/plugins/manual_move_plugin.py: a Plugin-like base + a subclass using
# the three real decoration shapes (bare, @property/@overrides, @staticmethod/@overrides).
class PluginBase:
    @property
    def endpoints(self) -> list:
        return []

    def on_message_receive(self) -> str:
        return "base"

    def to_dict(self) -> dict:
        return {}

    @staticmethod
    def from_dict(state: dict) -> dict:
        return state

    def load_state_dict(self, state: dict):
        pass


class MyPlugin(PluginBase):
    @property
    @overrides
    def endpoints(self) -> list:
        return [1]

    @overrides
    def on_message_receive(self) -> str:
        return "impl"

    @staticmethod
    @overrides
    def from_dict(state: dict) -> dict:
        return state

    @overrides
    def to_dict(self) -> dict:
        return {"a": 1}

    @overrides
    def load_state_dict(self, state: dict):
        pass


# --- core behaviour --------------------------------------------------------------------
def test_bare_overrides_keeps_method_and_marks_it():
    class Impl(Base):
        @overrides
        def run(self) -> str:
            return "impl"
    assert Impl().run() == "impl"
    assert Impl.run.__override__ is True


def test_docstring_inherited_when_override_has_none():
    class Impl(Base):
        @overrides
        def run(self) -> str:
            return "impl"
    assert Impl.run.__doc__ == "base docstring."


def test_own_docstring_is_preserved():
    class Impl(Base):
        @overrides
        def run(self) -> str:
            """own docstring."""
            return "impl"
    assert Impl.run.__doc__ == "own docstring."


def test_missing_base_method_raises_typeerror():
    with pytest.raises(TypeError, match="No super class method found"):
        class Impl(Base):
            @overrides
            def not_in_base(self) -> str:
                return "x"


def test_method_found_in_any_base_of_multiple_inheritance():
    class Impl(Base, Other):
        @overrides
        def extra(self) -> str:
            return "impl-extra"
    assert Impl().extra() == "impl-extra"
    assert Impl.extra.__override__ is True


def test_method_inherited_from_grandparent_is_found():
    class Child(Parent):
        @overrides
        def deep(self) -> str:
            return "child"
    assert Child().deep() == "child"


def test_dotted_base_class_is_resolved():
    # exercises _get_base_class's attribute walk: base referenced as collections.OrderedDict
    class MyDict(collections.OrderedDict):
        @overrides
        def keys(self):
            return super().keys()
    assert list(MyDict({"a": 1}).keys()) == ["a"]


def test_annotated_override_of_builtin_method_passes():
    """C-implemented bases (list.append, list.__iter__) report a positional-only, unannotated
    signature on Python 3.14+; there is nothing to compare an annotated Python override against,
    so only the existence check applies."""
    class Frozen(list):
        @overrides
        def append(self, object: int) -> None:  # pylint: disable=redefined-builtin
            raise ValueError("frozen")

        @overrides
        def __iter__(self) -> collections.abc.Iterator[int]:
            return super().__iter__()
    assert list(Frozen([1, 2])) == [1, 2]
    with pytest.raises(ValueError, match="frozen"):
        Frozen().append(1)


def test_annotation_key_renders_ellipsis_as_written():
    """`tuple[int, ...]` keys the same whether it is the live alias or the future-import string."""
    assert _ov_inner._annotation_key(tuple[int, ...]) == "tuple[int, ...]"
    assert _ov_inner._annotation_key(tuple[int, ...]) == _ov_inner._annotation_key("tuple[int, ...]", globals())


def test_future_import_variadic_tuple_matches_live_alias():
    """A future-imported base `-> tuple[int, ...]` and a non-future override with the same hint
    are the same signature. Exec'd without the future import, as in the dotted-type test."""
    ns = {"overrides": overrides, "VariadicBase": _VariadicBase}
    src = ("class Variadic(VariadicBase):\n"
           "    @overrides\n"
           "    def size(self) -> tuple[int, ...]:\n"
           "        return (1, 2)\n")
    exec(compile(src, "<variadic>", "exec", dont_inherit=True), ns)  # dont_inherit: no future import
    assert not isinstance(inspect.signature(ns["Variadic"].size).return_annotation, str)
    assert ns["Variadic"]().size() == (1, 2)


# --- decorator stacks used in the real code -------------------------------------------
def test_property_over_overrides_stack():
    # mirrors manual_move_plugin.endpoints: @property over @overrides on a base property
    class Impl(PluginBase):
        @property
        @overrides
        def endpoints(self) -> list:
            return [1]
    assert Impl().endpoints == [1]
    # the marker must land on the wrapped getter, else the shim never fired through @property
    assert Impl.endpoints.fget.__override__ is True


def test_staticmethod_over_overrides_stack():
    # mirrors server.SimState.from_dict: @staticmethod over @overrides on a base staticmethod
    class Impl(PluginBase):
        @staticmethod
        @overrides
        def from_dict(state: dict) -> dict:
            return dict(state)
    assert Impl.from_dict({"y": 2}) == {"y": 2}
    # the marker must land on the wrapped function under @staticmethod
    assert Impl.__dict__["from_dict"].__func__.__override__ is True


def test_realistic_plugin_subclass_builds_and_marks_overrides():
    # MyPlugin mirrors manual_move_plugin and is built at import — getting here means the
    # whole module imported, i.e. every @overrides above found its base method.
    p = MyPlugin()
    assert p.endpoints == [1]
    assert p.on_message_receive() == "impl"
    assert p.to_dict() == {"a": 1}
    assert MyPlugin.from_dict({"x": 1}) == {"x": 1}
    assert MyPlugin.on_message_receive.__override__ is True
    assert MyPlugin.to_dict.__override__ is True


# --- signature equality check ----------------------------------------------------------
def test_signature_mismatch_raises_typeerror():
    with pytest.raises(TypeError, match="check_signature"):
        class Impl(Base):
            @overrides
            def run(self, x: int) -> str:
                return "impl"


def test_return_type_mismatch_raises_typeerror():
    with pytest.raises(TypeError, match="return type"):
        class Impl(Base):
            @overrides
            def run(self) -> bytes:
                return b"impl"


def test_covariant_return_subclass_passes():
    """Mirrors RaceState.from_dict -> RaceState over Serializable.from_dict -> Serializable:
    the return type is a subclass of the base's -> Liskov-compatible, no flag needed."""
    assert CovRaceState.from_dict({}) is None


def test_covariant_return_self_referencing_passes():
    """Mirrors SimState.from_dict -> SimState: the return annotation names the class being
    defined (not yet in module globals during its own class body) — checked via the base."""
    assert CovSimState.from_dict({}) is None


def test_covariant_return_generic_alias_still_raises():
    """A generic-alias return (tuple[bytes, tuple]) is not a class -> never covariant."""
    with pytest.raises(TypeError, match="return type"):
        class MsgpackCodec(CovCodec):
            @overrides
            def encode(self, resp: dict) -> tuple[bytes, bytes]:
                return b"", b""


def test_future_import_dotted_type_matches_live_type():
    """Regression for ScreenDisplayerTkinter/SDL2: the future-imported base stores `frame: np.ndarray`
    as the string "np.ndarray"; a non-future override stores the live numpy.ndarray type. The override
    is exec'd without the future import because `from __future__ import annotations` is per-module and
    this test file needs it for its self-referencing covariance bases."""
    ns = {"np": np, "overrides": overrides, "DottedBackend": _DottedBackend}
    src = ("class DottedTk(DottedBackend):\n"
           "    @overrides\n"
           "    def update_frame(self, frame: np.ndarray):\n"
           "        pass\n")
    exec(compile(src, "<dotted>", "exec", dont_inherit=True), ns)  # dont_inherit: no future import
    assert not isinstance(inspect.signature(ns["DottedTk"].update_frame).parameters["frame"].annotation, str)
    assert ns["DottedTk"]().update_frame is not None


def test_annotation_key_resolves_dotted_string_like_live_type():
    """The two spellings of one type — the string 'np.ndarray' and the live numpy.ndarray — must key
    identically, and the dotted string must resolve through module globals."""
    assert _ov_inner._annotation_key("np.ndarray", globals()) == "ndarray"
    assert _ov_inner._annotation_key(np.ndarray, globals()) == "ndarray"
    assert _ov_inner._annotation_key("np.ndarray", globals()) == _ov_inner._annotation_key(np.ndarray, globals())


def test_annotation_mismatch_raises_typeerror():
    class Annotated(Base):
        @overrides
        def run(self) -> str:
            return "impl"

    with pytest.raises(TypeError, match="annotation"):
        class Impl(Annotated):
            @overrides
            def run(self, x: int) -> str:  # base has no params -> mismatch
                return "impl"


def test_default_mismatch_raises_typeerror():
    class WithDefault(Base):
        def to_dict(self, depth: int = 2) -> dict:
            return {}

    with pytest.raises(TypeError, match="default"):
        class Impl(WithDefault):
            @overrides
            def to_dict(self, depth: int = 3) -> dict:  # default changed
                return {}


def test_exact_match_inherited_from_grandparent_passes():
    class Child(Parent):
        @overrides
        def deep(self) -> str:
            return "child"
    assert Child().deep() == "child"


def test_check_signature_false_skips_check_and_tags_method():
    class Impl(Base):
        @overrides(check_signature=False)
        def run(self, x: int) -> bytes:
            return b"impl"
    assert Impl().run(1) == b"impl"
    assert Impl.run.__changed_parameters__ is True


def test_check_signature_false_skips_annotation_and_return_check():
    class Annotated(Base):
        @overrides
        def run(self) -> str:
            return "impl"

    class Impl(Annotated):
        @overrides(check_signature=False)
        def run(self, x: int) -> bytes:  # would fail bare; flag opts out
            return b"impl"
    assert Impl().run(1) == b"impl"


def test_override_checks_immediate_base_not_root():
    """Base -> Derived1(check_signature=False) -> Derived2: Derived2 must compare against
    Derived1's signature, not Base's. Derived2 adds a param on top of Derived1's change."""
    assert Derived2().run(1) == "d2:1"
    assert not hasattr(Derived2.run, "__changed_parameters__")


def test_override_immediate_base_mismatch_still_raises():
    with pytest.raises(TypeError, match="check_signature"):
        class DerivedBad(_DerivedChanged):
            @overrides
            def run(self, x: int, y: int) -> str:  # mismatches Derived1
                return "d2"


# --- called form: check_signature is the only kwarg ------------------------------------
def test_called_form_empty_parens():
    class Impl(Base):
        @overrides()
        def run(self) -> str:
            return "impl"
    assert Impl().run() == "impl"


def test_called_form_check_signature_false():
    class Impl(Base):
        @overrides(check_signature=False)
        def run(self, x: int) -> bytes:
            return b"impl"
    assert Impl().run(1) == b"impl"
    assert Impl.run.__changed_parameters__ is True


def test_called_form_check_signature_true_still_checks():
    class Impl(Base):
        @overrides(check_signature=True)
        def run(self) -> str:
            return "impl"
    assert Impl().run() == "impl"


def test_called_form_still_validates_existence():
    with pytest.raises(TypeError, match="No super class method found"):
        class Impl(Base):
            @overrides(check_signature=False)
            def not_in_base(self) -> str:
                return "x"


# --- @override alias -------------------------------------------------------------------
def test_override_is_alias_of_overrides():
    assert override is overrides


def test_override_alias_behaves_identically():
    class Impl(Base):
        @override
        def run(self) -> str:
            return "impl"
    assert Impl().run() == "impl"
    assert Impl.run.__override__ is True


# --- trim guarantees -------------------------------------------------------------------
def test_public_surface_is_exactly_overrides_and_override():
    assert set(_ov.__all__) == {"overrides", "override"}


@pytest.mark.parametrize("removed", [
    "EnforceOverrides", "final", "enforce", "signature", "typing_utils", "verify_signature",
])
def test_signature_engine_and_enforce_stay_trimmed(removed):
    assert not hasattr(_ov, removed)
