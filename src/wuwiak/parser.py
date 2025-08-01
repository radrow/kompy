from pathlib import Path
import typing

import parsy as P

from . import error
from . import ast
from . import lexer as L
from .lexer import lex


# ==============================================================================
# General

class ParseError(error.WuwiakError):
    def __init__(self, pe):
        self.parsy_error = pe


def parse_file(filepath: Path | str, name: typing.Optional[str] = None) -> ast.Program:
    filepath = Path(filepath)
    content = filepath.read_text(encoding='utf-8')
    name = name if name else filepath.stem
    try:
        return program(module_name=name).parse(content)
    except P.ParseError as e:
        raise ParseError from e


def parens(pars):
    return lex(L.lparen) >> pars << lex(L.rparen)


def bracs(pars):
    return lex(L.lbrac) >> pars << lex(L.rbrac)


# ==============================================================================
# Expression

expr = P.forward_declaration()

expr_var = L.ident.map(lambda v: ast.Var(name=v))

expr_integer = L.integer.map(lambda v: ast.Int(v=v))

expr_boolean = P.alt(
    L.kw("true").result(ast.Bool(v=True)),
    L.kw("false").result(ast.Bool(v=False)),
)

expr_string = lex(L.string).map(lambda v: ast.String(v=v))


@P.generate
def expr_call():
    fname = yield L.ident
    args = yield parens(expr.sep_by(L.token(',')))
    return ast.Call(fun=fname, args=args)


expr_atom = P.alt(
    expr_call,
    expr_var,
    expr_integer,
    expr_boolean,
    expr_string,
    parens(expr)
)

expr_unop = P.seq(
    fun=L.unop,
    args=expr_atom.map(lambda e: [e])
).combine_dict(ast.Call)

expr_simple = P.alt(
    expr_atom,
    expr_unop,
)


def prec(o):
    if o in ["||", "&&"]:
        return 0
    if o in ["==", "!=", "<", "<=", ">", ">="]:
        return 10
    if o in "+-":
        return 20
    if o in "*/%":
        return 30
    raise ValueError(f"Wtf operator {o}")


def op_bind(o):
    if o in ["||", "&&"]:
        return 'R'
    if o in "*/%+-":
        return 'L'
    if o in ["==", "!=", "<", "<=", ">", ">="]:
        return 'R'  # TODO this should be `None`
    raise ValueError(f"Wtf operator {o}")


@P.generate
def expr_op():
    # We first parse everything flat
    expr_seq = yield (P.seq(expr_simple, L.op)).at_least(1)
    last = yield expr

    # Convert to postfix notation. Strings represent operators.
    def is_op(x):
        return isinstance(x, str)

    stack = []
    postfix = []
    for (e, o) in expr_seq:
        postfix.append(e)

        while (stack and
               ((op_bind(o) == 'L' and prec(o) <= prec(stack[-1])) or
                (op_bind(o) == 'R' and prec(o) < prec(stack[-1]))
                )
               ):
            postfix.append(stack.pop())

        stack.append(o)

    postfix.append(last)

    while stack:
        postfix.append(stack.pop())

    # Build tree
    stack = []
    for entry in postfix:
        if is_op(entry):
            op_r = stack.pop()
            op_l = stack.pop()
            op_e = ast.Call(fun=entry, args=[op_l, op_r])
            stack.append(op_e)
        else:
            stack.append(entry)

    assert len(stack) == 1
    return stack[0]


expr.become(P.alt(
    expr_op,
    expr_simple,
).desc("expression"))


# ==============================================================================
# Type

typ = P.forward_declaration()

type_var = L.ident.map(lambda v: ast.TypeVar(name=v))

typ.become(
    P.alt(
        type_var,
    ).desc("type")
)


# ==============================================================================
# Statement

stmt = P.forward_declaration()
block = P.forward_declaration()

stmt_var_decl = P.seq(
    typ=typ,
    name=L.ident,
    value=(L.token('=') >> expr).optional(),
    _semi=L.semicolon,
).combine_dict(ast.VarDecl)

stmt_expr = P.seq(
    expr=expr,
    _semi=L.semicolon,
).combine_dict(ast.SExpr)

stmt_assg = P.seq(
    name=L.ident,
    _eq=L.token('='),
    value=expr,
    _semi=L.semicolon,
).combine_dict(ast.Assg)

stmt_return = P.seq(
    _return=L.kw('return'),
    expr=expr.optional(),
    _semi=L.semicolon,
).combine_dict(ast.Return)

stmt_if = P.seq(
    _if=L.kw('if'),
    cond=expr,
    then_block=block,
    else_block=(L.kw('else') >> block).optional(),
).combine_dict(ast.If)

stmt_dopóty_dopóki = P.seq(
    _dopóty=L.kw('dopóty dopóki'),
    cond=expr,
    body=block,
).combine_dict(ast.DopótyDopóki)

stmt.become(P.alt(
    stmt_var_decl,
    stmt_assg,
    stmt_return,
    stmt_expr,
    stmt_if,
    stmt_dopóty_dopóki,
).desc("statement"))

block.become(bracs(
    stmt.many()
).map(lambda stmts: ast.Block(stmts=stmts)))


# ==============================================================================
# Top-level declaration

top_fundecl = P.seq(
    ret=typ,
    name=L.ident,
    args=parens((P.seq(typ, L.ident)).sep_by(L.token(","))),
    body=block,
).combine_dict(ast.FunDecl)


def program(module_name: str):
    return L.skip >> top_fundecl.many().map(lambda decls: ast.Program(decls=decls, name=module_name))
