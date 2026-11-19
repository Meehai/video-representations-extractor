"""First-party micro-shim providing the `@overrides` / `@override` decorator.

This is **not** the third-party `overrides` package — it is a small in-house existence check that
keeps robosim's "zero external pure-Python deps" goal. It asserts at class-definition time that the
decorated method really overrides a same-named method in some base class, raising `TypeError` if
not, and that the override's *signature* matches the base's — parameter names/kinds/order/defaults
plus the annotations and return type, compared as written (no type resolution, no subtyping
engine). An override that legitimately changes the signature opts out with
`@overrides(check_signature=False)` — the same keyword the third-party `overrides` package uses, so
source written against either resolves identically in this conda env — which also tags the method
(`__changed_parameters__`).

Why no typing engine: upstream's ~780-line signature check resolves annotations to real types and
runs a full variance/subtyping analysis (contravariant params, covariant returns, generic
variance). Here we compare the hints *textually* — a type object becomes its `__qualname__`, a
string is resolved through its module globals when it names a type (dotted paths like `np.ndarray`
included, so a future-imported `'np.ndarray'` and a live `numpy.ndarray` compare equal) and
otherwise stays as-written, and a generic/union alias renders like its source form (`list[X]`,
`X | None`). That is O(annotations) per decorated method at import time and catches "I changed
`bytes` to `tuple[bytes, bytes]`" drift without resolving anything. Runtime type enforcement is
TYPEGUARD's job, not this shim's.

The base-class detection mirrors upstream overrides 7.7.0 exactly: the decorator runs inside the
class body (before the class object exists), so it walks the class-body frame's bytecode to recover
the base-class names, resolves them through the defining module's namespace, and checks each for the
method name. The `overrides -> _overrides -> sys._getframe(3)` call depth is kept identical to
upstream — do not flatten it, the frame offset depends on it.
"""
from __future__ import annotations
import dis
import functools
import inspect
import sys
import types
from types import FrameType
from typing import Callable, Optional, Union


def overrides(method: Optional[Callable] = None, *, check_signature: bool = True) -> Callable:
    """Mark a method as overriding a base-class method of the same name.

    Usable bare (`@overrides`) or called (`@overrides(check_signature=False)`). By default the
    override's signature (param names/kinds/order/defaults + annotations + return type) must match
    the base's, as written; `check_signature=False` skips that check and tags the method with
    `__changed_parameters__`. The keyword matches the third-party `overrides` package (which ships
    `check_signature`/`check_at_runtime`) so the same source runs against either.
    """
    if method is not None:
        return _overrides(method, check_signature=check_signature)
    return functools.partial(overrides, check_signature=check_signature)


override = overrides  # upstream exposes both spellings; identical behaviour here


def _overrides(method: Callable, *, check_signature: bool) -> Callable:
    setattr(method, "__override__", True)
    if not check_signature:
        setattr(method, "__changed_parameters__", True)
    global_vars = getattr(method, "__globals__", None) or vars(sys.modules[method.__module__])
    for super_class in _get_base_classes(sys._getframe(3), global_vars):
        if hasattr(super_class, method.__name__):
            base_method = _base_callable(super_class, method.__name__)
            if not method.__doc__:  # inherit the base method's docstring when we have none
                method.__doc__ = getattr(base_method, "__doc__", None)
            if check_signature:
                _check_signature(method, base_method, super_class)
            return method
    raise TypeError(f"{method.__qualname__}: No super class method found")


def _base_callable(super_class, name: str):
    """The callable signature should be compared against: unwraps `property` to its getter."""
    attr = getattr(super_class, name)
    return attr.fget if isinstance(attr, property) else attr


def _check_signature(override: Callable, base: Callable, super_class):
    """Compare override vs base signature (params + annotations + return type), as written.

    Skips silently when the base is C-implemented (`list.append`, `dict.keys`: positional-only and
    unannotated, so nothing to compare) or can't be introspected — the existence check above
    already passed, that's all we can assert there.
    """
    if inspect.ismethoddescriptor(base) or inspect.isbuiltin(base):
        return
    try:
        override_sig = inspect.signature(override)
        base_sig = inspect.signature(base)
    except (TypeError, ValueError):
        return

    override_globals = getattr(override, "__globals__", None)
    base_globals = getattr(base, "__globals__", None)

    override_params = list(override_sig.parameters.values())
    base_params = list(base_sig.parameters.values())
    if len(override_params) != len(base_params):
        raise TypeError(
            f"{override.__qualname__}: {len(override_params)} params vs {len(base_params)} in the "
            f"base ({override_sig} vs {base_sig}); use check_signature=False to override")
    for o_param, b_param in zip(override_params, base_params):
        if o_param.name != b_param.name or o_param.kind is not b_param.kind:
            raise TypeError(
                f"{override.__qualname__}: param {o_param.name!r} differs from the base's "
                f"{b_param.name!r} ({override_sig} vs {base_sig}); use check_signature=False")
        if not _defaults_match(o_param.default, b_param.default):
            raise TypeError(
                f"{override.__qualname__}: default of {o_param.name!r} differs from the base "
                f"({override_sig} vs {base_sig}); use check_signature=False")
        if _annotation_key(o_param.annotation, override_globals) != _annotation_key(b_param.annotation, base_globals):
            raise TypeError(
                f"{override.__qualname__}: annotation of {o_param.name!r} differs from the base "
                f"({override_sig} vs {base_sig}); use check_signature=False")
    if _annotation_key(override_sig.return_annotation, override_globals) != _annotation_key(base_sig.return_annotation, base_globals):
        if not _is_covariant_return(override, base, override_sig.return_annotation,
                                    base_sig.return_annotation, super_class):
            raise TypeError(
                f"{override.__qualname__}: return type differs from the base "
                f"({override_sig} vs {base_sig}); use check_signature=False")


def _is_covariant_return(override: Callable, base: Callable, override_ret, base_ret,
                         super_class) -> bool:
    """True when the override's return type is a subclass of the base's (covariant narrowing —
    Liskov-compatible, so no flag needed).

    Resolves each annotation to an actual class: the override's via its module globals, except a
    self-referencing return (e.g. `RaceState.from_dict -> RaceState`, where the class name is not
    in globals yet during its own class body) which is checked through the base `super_class` this
    method was found on — if `super_class` is the base return type or a subclass of it, so is the
    override's class. Alias/union hints (e.g. `tuple[bytes, tuple]`) are not classes and never
    count as covariant.
    """
    base_cls = _resolve_as_type(base_ret, getattr(base, "__globals__", None))
    if base_cls is None:
        return False

    override_cls = _resolve_as_type(override_ret, getattr(override, "__globals__", None))
    if override_cls is not None:
        try:
            return issubclass(override_cls, base_cls)
        except TypeError:
            return False

    # self-referencing return: the annotation names the class being defined, which is not in
    # globals yet. It subclasses `super_class` (the base this method was found on); if that base
    # is the base return type (or a subclass of it), the override's class is too.
    override_class_name = override.__qualname__.rsplit(".", 1)[0]
    if isinstance(override_ret, str) and override_ret == override_class_name:
        try:
            return issubclass(super_class, base_cls)
        except TypeError:
            return False
    return False


def _resolve_as_type(annotation, globals_):
    """Resolve an annotation to an actual class for the covariance check: a `type` is itself; a
    string name — bare (`RaceState`) or a dotted module path (`np.ndarray`) — is looked up through
    the defining module's globals; anything else (generic alias, union) is not a class and can't be
    a covariant return."""
    if isinstance(annotation, type):
        return annotation
    if isinstance(annotation, str) and globals_:
        obj = globals_.get(annotation)
        if obj is None and "." in annotation:
            parts = annotation.split(".")
            obj = globals_.get(parts[0])
            for comp in parts[1:]:
                obj = getattr(obj, comp, None)
                if obj is None:
                    break
        return obj if isinstance(obj, type) else None
    return None


def _defaults_match(a, b) -> bool:
    if a is b:
        return True
    if a is inspect.Parameter.empty or b is inspect.Parameter.empty:
        return a is b
    try:
        return a == b
    except Exception:  # noqa: BLE001 - a non-comparable default is a mismatch
        return False


def _annotation_key(annotation, globals_=None):
    """A comparable key for one annotation, independent of `from __future__ import annotations`
    and of module qualification: a type object becomes its __qualname__; a generic alias
    (`list[X]`, `tuple[bytes, tuple]`) and a union (`X | None`) render the same way they are
    written in source. A string annotation is first resolved through the defining module's globals
    (dotted paths like `np.ndarray` included): a name that names a type keys as that type's
    __qualname__, so a future-imported base (`'np.ndarray'`) and a non-future override
    (`numpy.ndarray`) compare equal; anything unresolvable stays as-written."""
    if annotation is inspect.Parameter.empty:
        return None
    if annotation is Ellipsis:  # tuple[int, ...]: render as written, not as `Ellipsis`
        return "..."
    if isinstance(annotation, str):
        annotation = _resolve_as_type(annotation, globals_) or annotation
    if isinstance(annotation, type):
        return "None" if annotation is type(None) else annotation.__qualname__
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)
    if isinstance(annotation, types.UnionType) or (origin is Union and args):  # `X | Y`: no __origin__ before 3.14
        return " | ".join(_annotation_key(arg) for arg in args)
    if origin is not None and args:
        return f"{_annotation_key(origin)}[{', '.join(_annotation_key(arg) for arg in args)}]"
    return str(annotation)


def _get_base_classes(frame: FrameType, namespace: dict) -> list:
    return [_get_base_class(names, namespace) for names in _get_base_class_names(frame)]


def _get_base_class_names(frame: FrameType) -> list:
    """Recover base-class names from the class-body frame's bytecode (LOAD_NAME/GLOBAL/ATTR)."""
    current_item: list = []
    items: list = []
    add_last_step = True
    for instruction in dis.get_instructions(frame.f_code):
        if instruction.offset > frame.f_lasti:
            break
        if instruction.opcode not in dis.hasname:
            continue
        if not add_last_step:
            items = []
            add_last_step = True
        if instruction.opname in ("LOAD_NAME", "LOAD_GLOBAL"):
            if current_item:
                items.append(current_item)
            current_item = [instruction.argval]
        elif instruction.opname == "LOAD_ATTR" and current_item:
            current_item.append(instruction.argval)
        else:
            if current_item:
                items.append(current_item)
            current_item = []
            add_last_step = False
    if current_item:
        items.append(current_item)
    return items


def _get_base_class(components: list, namespace: dict):
    try:
        obj = namespace[components[0]]
    except KeyError:
        builtins = namespace["__builtins__"]
        obj = builtins[components[0]] if isinstance(builtins, dict) else getattr(builtins, components[0])
    for component in components[1:]:
        if hasattr(obj, component):
            obj = getattr(obj, component)
    return obj
