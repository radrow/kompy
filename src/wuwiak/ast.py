"""
Abstract Syntax Tree.
"""
import typing
from attrs import frozen, field


@frozen
class Node:
    """
    Generic AST node
    """


# ==============================================================================
# Expressions

@frozen(auto_attribs=True, kw_only=True)
class Expr(Node):
    """
    Expressions — anything that can be evaluated into a value
    """
    type: 'typing.Optional[Type]' = field(default=None)


@frozen(auto_attribs=True, kw_only=True)
class Var(Expr):
    """
    Variable
    """
    name: str


@frozen(auto_attribs=True, kw_only=True)
class Int(Expr):
    """
    Integer literal
    """
    v: int


@frozen(auto_attribs=True, kw_only=True)
class Bool(Expr):
    """
    Boolean literal (`true`/`false`)
    """
    v: bool


@frozen(auto_attribs=True, kw_only=True)
class String(Expr):
    """
    String literal
    """
    v: str


@frozen(auto_attribs=True, kw_only=True)
class Call(Expr):
    """
    Function call
    """
    fun: str
    args: typing.List[Expr]


# ==============================================================================
# Types

@frozen()
class Type(Node):
    """
    Type of an expression
    """


@frozen(auto_attribs=True, kw_only=True)
class TypeVar(Type):
    """
    Constant type variable, such as `int`
    """
    name: str


@frozen(auto_attribs=True, kw_only=True)
class TypeFun(Type):
    """
    Functional type. Not available in the syntax (thus technically
    shouldn't be here).
    """
    args: typing.List[Type]
    ret: Type


# ==============================================================================
# Statements

@frozen()
class Stmt(Node):
    """
    Statement
    """


@frozen(auto_attribs=True)
class Block(Node):
    """
    Block is a list of statements enclosed with `{ }`
    """
    stmts: typing.List[Stmt]


@frozen(auto_attribs=True, kw_only=True)
class SExpr(Stmt):
    """
    Plain expression in a statement position. E.g. `print_string("hello")`
    """
    expr: Expr


@frozen(auto_attribs=True, kw_only=True)
class VarDecl(Stmt):
    """
    Variable declaration with optional assignment
    """
    typ: Type
    name: str
    value: typing.Optional[Expr] = None


@frozen(auto_attribs=True, kw_only=True)
class Assg(Stmt):
    """
    Variable assignment
    """
    name: str
    value: Expr


@frozen(auto_attribs=True, kw_only=True)
class Return(Stmt):
    """
    Return statement
    """
    expr: typing.Optional[Expr] = None


@frozen(auto_attribs=True, kw_only=True)
class If(Stmt):
    """
    If statement with optional `else` clause
    """
    cond: Expr
    then_block: Block
    else_block: typing.Optional[Block] = None

@frozen(auto_attribs=True, kw_only=True)
class DopótyDopóki(Stmt):
    """
    Pętla dopóty-dopóki
    """
    cond: Expr
    body: Block


# ==============================================================================
# Top-level declarations

@frozen()
class TopDecl(Node):
    """
    Top-level declaration
    """


@frozen(auto_attribs=True, kw_only=True)
class FunDecl(TopDecl):
    """
    Function declaration
    """
    name: str
    args: typing.List[typing.Tuple[Type, str]]
    ret: Type
    body: Block


@frozen(auto_attribs=True, kw_only=True)
class Program(Node):
    """
    Program consists of top-level declarations
    """
    name: str
    decls: typing.List[TopDecl]
