import typing
import attr


@attr.s(frozen=True)
class Node:
    pass


# ==============================================================================
# Expressions

@attr.s(frozen=True)
class Expr(Node):
    type: typing.Any = attr.ib(default=None, kw_only=True)


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Var(Expr):
    name: str


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Int(Expr):
    v: int

    def __int__(self):
        return self.v


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Bool(Expr):
    v: bool

    def __bool__(self):
        return self.v


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class String(Expr):
    v: str

    def __str__(self):
        return self.v


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Call(Expr):
    fun: str
    args: typing.List[Expr]


# ==============================================================================
# Types

@attr.s(frozen=True)
class Type(Node):
    pass


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class TypeVar(Type):
    name: str


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class TypeFun(Type):
    args: typing.List[Type]
    ret: Type


# ==============================================================================
# Statements

@attr.s(frozen=True)
class Stmt(Node):
    pass


@attr.s(frozen=True, auto_attribs=True)
class Block(Node):
    stmts: typing.List[Stmt]


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class SExpr(Stmt):
    expr: Expr


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class VarDecl(Stmt):
    typ: Type
    name: str
    value: typing.Optional[Expr] = None


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Assg(Stmt):
    name: str
    value: Expr


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class Return(Stmt):
    expr: typing.Optional[Expr] = None


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class If(Stmt):
    cond: Expr
    then_block: Block
    else_block: typing.Optional[Block] = None


# ==============================================================================
# Top-level declarations

@attr.s(frozen=True)
class TopDecl(Node):
    pass


@attr.s(frozen=True, auto_attribs=True, kw_only=True)
class FunDecl(TopDecl):
    name: str
    args: typing.List[typing.Tuple[str, Type]]
    ret: Type
    body: Block
