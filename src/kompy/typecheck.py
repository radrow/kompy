import attr

from . import ast


@attr.s(frozen=True, auto_attribs=True, kw_only=False)
class TypecheckError(Exception):
    msg: str


def typed(node, typ):
    return attr.evolve(node, type=typ)


t_int = ast.TypeVar(name='int')

t_bool = ast.TypeVar(name='bool')

t_string = ast.TypeVar(name='string')

t_void = ast.TypeVar(name='void')

return_t = '$RETURN'

def t_fun(args, ret):
    try:
        args = list(args)
    except TypeError:
        args = [args]

    return ast.TypeFun(args=args, ret=ret)


def init_env():
    return {
        '+': t_fun([t_int, t_int], t_int),
        '-': t_fun([t_int, t_int], t_int),
        '*': t_fun([t_int, t_int], t_int),
        '/': t_fun([t_int, t_int], t_int),
        '%': t_fun([t_int, t_int], t_int),
        '>=': t_fun([t_int, t_int], t_bool),
        '>': t_fun([t_int, t_int], t_bool),
        '<': t_fun([t_int, t_int], t_bool),
        '<=': t_fun([t_int, t_int], t_bool),
        '||': t_fun([t_bool, t_bool], t_bool),
        '&&': t_fun([t_bool, t_bool], t_bool),
        'print_int': t_fun(t_int, t_void)
    }


def get_var(env, name):
    if name not in env:
        raise TypecheckError(f"Undefined var: {name}")
    return env[name]


def get_return(env):
    return env[return_t]


def match_type(t_super, t_sub):
    """
    Whether `t0` is a supertype of `t1`.
    """
    if t_super != t_sub:
        raise TypecheckError(f"Type error: expected {t_super}, got {t_sub}")


def tc_expr(env, expr):
    match expr:
        case ast.Var(name=v):
            return typed(expr, get_var(env, v))
        case ast.Int():
            return typed(expr, t_int)

        case ast.Bool():
            return typed(expr, t_bool)

        case ast.String():
            return typed(expr, t_string)

        case ast.Call(fun=fun, args=(op_l, op_r)) if fun in ('==', '!='):
            # This is a special case where we allow "polymorphism"
            op_l_t = tc_expr(env, op_l)
            op_r_t = tc_expr(env, op_r)
            match_type(op_l_t.type, op_r_t.type)
            return attr.evolve(
                expr,
                args=(op_l_t, op_r_t),
                type=t_bool
            )

        case ast.Call(fun=fun, args=args):
            fun_t = get_var(env, fun)
            match fun_t:
                case ast.TypeVar():
                    raise TypecheckError(f"Not a function: {fun_t}")
                case ast.TypeFun(args=arg_types, ret=ret):
                    if len(args) != len(arg_types):
                        raise TypecheckError(f"Argument number mismatch")
                    args_t = []
                    for (arg, arg_type) in zip(args, arg_types):
                        arg_t = tc_expr(env, arg)
                        match_type(arg_type, arg_t.type)
                        args_t.append(arg_t)
                    return attr.evolve(
                        expr,
                        args=args_t,
                        type=ret
                    )


def tc_block(env, block):
    env = env.copy()

    block_t = []

    for stmt in block.stmts:
        match stmt:
            case ast.SExpr(expr=expr):
                expr_t = tc_expr(env, expr)
                stmt_t = attr.evolve(
                    stmt,
                    expr=expr_t
                )
            case ast.Return(expr=expr):
                expr_t = tc_expr(env, expr)
                match_type(get_return(env), expr_t.type)
                stmt_t = attr.evolve(
                    stmt,
                    expr=expr_t
                )
            case ast.If(cond=cond, then_block=then_block, else_block=else_block):
                cond_t = tc_expr(env, cond)
                match_type(t_bool, cond_t.type)
                then_block_t = tc_block(env, then_block)

                else_block_t = None
                if else_block:
                    else_block_t = tc_block(env, else_block)

                stmt_t = attr.evolve(
                    stmt,
                    cond=cond_t,
                    then_block=then_block_t,
                    else_block=else_block_t
                )
            case ast.VarDecl(typ=typ, name=name, value=value):
                value_t = None
                if value:
                    # We check the value before we update the env to handle
                    # things like `int x = x || true`.
                    value_t = tc_expr(env, value)
                    match_type(typ, value_t.type)
                env[name] = typ
                stmt_t = attr.evolve(
                    stmt,
                    value=value_t
                )
            case ast.Assg(name=name, value=value):
                value_t = tc_expr(env, value)
                var_type = get_var(env, name)
                match_type(var_type, value_t.type)
                stmt_t = attr.evolve(
                    stmt,
                    value=value_t
                )
        block_t.append(stmt_t)

    block_t = ast.Block(stmts=block_t)
    return block_t


def tc_decl(env, decl):
    env = env.copy()

    match decl:
        case ast.FunDecl(
                ret=ret,
                args=args,
                body=body,
        ):
            env[return_t] = ret
            for (arg_t, arg_n) in args:
                env[arg_n] = arg_t
            body_t = tc_block(env, body)
            return attr.evolve(
                decl,
                body=body_t
            )


def tc_src(env, decls):
    # First, add all function signatures
    for decl in decls:
        match decl:
            case ast.FunDecl(ret=ret, name=name, args=args):
                env[name] = ast.TypeFun(
                    ret=ret,
                    args=[arg_t for arg_t, _ in args]
                )
            case _:
                pass

    # Typecheck all declarations in the updated env
    decls_t = []
    for decl in decls:
        decl_t = tc_decl(env, decl)
        decls_t.append(decl_t)

    return decls_t
