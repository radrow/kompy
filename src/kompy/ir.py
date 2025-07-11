import typing
import attr

from . import ast
from . import typecheck as t



# ==============================================================================
# Immediates

@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Atom:
    """
    Variable or an immediately representable value
    """
    pass


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Var(Atom):
    name: str


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Int(Atom):
    v: int


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class String(Atom):
    v: str



# ==============================================================================
# Instructions


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Instr:
    pass


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Push(Instr):
