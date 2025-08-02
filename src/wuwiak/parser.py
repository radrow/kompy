from pathlib import Path
import typing

import attrs
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

kod = ""

def parse_file(filepath: Path | str, name: typing.Optional[str] = None) -> ast.Program:
    global kod
    filepath = Path(filepath)
    content = filepath.read_text(encoding='utf-8')
    kod = content
    name = name if name else filepath.stem
    try:
        return program(module_name=name).parse(content)
    except P.ParseError as e:
        raise ParseError(e)


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
    skip = 0
    for i, entry in enumerate(postfix):
        if skip > 0:
            skip -= 1
            continue
        if is_op(entry):
            first = True
            equaled = False
            while i + skip != len(postfix) and postfix[i + skip] == '==':
                equaled = True
                if first:
                    first = False
                    op_r = stack.pop()
                    op_l = stack.pop()
                    op_e = ast.Call(fun=entry, args=[op_l, op_r])
                else:
                    op_rr = stack.pop()
                    op_e = ast.Call(
                        fun='&&',
                        args=[
                            op_e,
                            ast.Call(fun=entry, args=[op_r, op_rr])
                        ])
                skip += 1

            if not equaled:
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

if_head = P.seq(
    L.kw('if'),
    expr,
).map(lambda e: e[1])

stmt_if = P.seq(
    cond=if_head.mark(),
    then_block=block,
    else_block=(L.kw('else') >> block).optional(),
).map(lambda s: ast.If(cond=s['cond'][1],
                       then_block=s['then_block'],
                       else_block=s['else_block'],
                       start=s['cond'][0],
                       end=s['cond'][2]
                       )
      )

dopóty_dopóki_head = P.seq(
    L.kw('dopóty dopóki'),
    expr,
).map(lambda e: e[1])

stmt_dopóty_dopóki = P.seq(
    cond=dopóty_dopóki_head.mark(),
    body=block,
).map(lambda s: ast.DopótyDopóki(cond=s['cond'][1],
                       body=s['body'],
                       start=s['cond'][0],
                       end=s['cond'][2]
                       )
      )

stmt_kurwa = P.seq(
    _return=L.kw('kurwa XD'),
    _semi=L.semicolon,
).combine_dict(ast.Kurwa)

stmt_chuj = P.seq(
    _return=L.kw('albo chuj'),
    _semi=L.semicolon,
).combine_dict(ast.Chuj)

def with_loc(s):
    return s.mark().map(lambda s: attrs.evolve(s[1], start=s[0], end=s[2]))

stmt.become(P.alt(
    with_loc(stmt_kurwa),
    with_loc(stmt_chuj),
    with_loc(stmt_var_decl),
    with_loc(stmt_assg),
    with_loc(stmt_return),
    with_loc(stmt_expr),
    stmt_dopóty_dopóki,
    stmt_if,
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
