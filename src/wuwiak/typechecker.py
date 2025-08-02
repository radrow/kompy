"""
Decorating AST with type annotations
"""
import attrs

from . import ast
from . import error

@attrs.frozen(auto_attribs=True, kw_only=False)
class TypecheckError(error.WuwiakError):
    """
    Exception class for user errors
    """
    msg: str


def typed(node, typ):
    """
    Decorates an expression with a type annotation
    """
    return attrs.evolve(node, type=typ)


t_int = ast.TypeVar(name='int')

t_bool = ast.TypeVar(name='bool')

t_string = ast.TypeVar(name='string')

t_void = ast.TypeVar(name='void')

t_any = ast.TypeVar(name='$any')

RETURN_VAR = '$RETURN'
KURWA_VAR = '$KURWA'
CHUJ_VAR = '$CHUJ'


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
        '!': t_fun([t_bool], t_bool),
        'print_int': t_fun(t_int, t_void),
        'print_bool': t_fun(t_bool, t_void),
        'print_string': t_fun(t_string, t_void),
        'print': t_fun(t_any, t_void),
        'itos': t_fun(t_int, t_string),
        KURWA_VAR: False,
        CHUJ_VAR: False,
    }


def get_var(env, name):
    if name not in env:
        raise TypecheckError(f"Undefined var: {name}")
    return env[name]


def get_return(env):
    return env[RETURN_VAR]


def match_type(t_super, t_sub):
    """
    Whether `t0` is a supertype of `t1`.
    """
    if t_super == t_any:
        return

    if t_super != t_sub:
        raise TypecheckError(f"Type error: expected {t_super}, got {t_sub}")


def tc_expr(env, expr):
    """
    Typechecks an expression
    """
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
            return attrs.evolve(
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
                        raise TypecheckError(f"Argument number mismatch: expected {len(arg_types)}, got {len(args)}")
                    args_t = []
                    for (arg, arg_type) in zip(args, arg_types):
                        arg_t = tc_expr(env, arg)
                        match_type(arg_type, arg_t.type)
                        args_t.append(arg_t)

                        # Dispatch polymorphic print
                        if fun == 'print':
                            match args_t[0].type:
                                case ast.TypeVar(name='int'):
                                    fun = 'print_int'
                                case ast.TypeVar(name='string'):
                                    fun = 'print_string'
                                case ast.TypeVar(name='bool'):
                                    fun = 'print_bool'
                                case _:
                                    raise TypecheckError(f"Unprintable type {args_t[0].type}")

                    return attrs.evolve(
                        expr,
                        fun=fun,
                        args=args_t,
                        type=ret
                    )


def tc_block(env, tail, block):
    """
    Incrementally typechecks a block of statements
    """
    env = env.copy()

    block_t = []

    for i, stmt in enumerate(block.stmts):
        # Check if in tail position
        tail = tail and i == len(block.stmts) - 1

        match stmt:
            case ast.SExpr(expr=expr):
                if tail:
                    raise TypecheckError("Missing return")
                expr_t = tc_expr(env, expr)
                stmt_t = attrs.evolve(
                    stmt,
                    expr=expr_t
                )
            case ast.Return(expr=expr):
                expr_t = tc_expr(env, expr)
                match_type(get_return(env), expr_t.type if expr_t else t_void)
                stmt_t = attrs.evolve(
                    stmt,
                    expr=expr_t
                )
            case ast.If(cond=cond, then_block=then_block, else_block=else_block):
                cond_t = tc_expr(env, cond)
                match_type(t_bool, cond_t.type)
                if else_block and not env[KURWA_VAR]:
                    env[KURWA_VAR] = True
                    then_block_t = tc_block(env, tail, then_block)
                    env[KURWA_VAR] = False
                else:
                    then_block_t = tc_block(env, tail, then_block)

                else_block_t = None
                if else_block:
                    if not env[CHUJ_VAR]:
                        env[CHUJ_VAR] = True
                        else_block_t = tc_block(env, tail, else_block)
                        env[CHUJ_VAR] = False
                    else:
                        else_block_t = tc_block(env, tail, else_block)
                elif tail:
                    raise TypecheckError("Missing return")

                stmt_t = attrs.evolve(
                    stmt,
                    cond=cond_t,
                    then_block=then_block_t,
                    else_block=else_block_t
                )
            case ast.DopótyDopóki(cond=cond, body=body):
                cond_t = tc_expr(env, cond)
                match_type(t_bool, cond_t.type)
                body_t = tc_block(env, tail, body)

                if tail:
                    raise TypecheckError("Missing return")

                stmt_t = attrs.evolve(
                    stmt,
                    cond=cond_t,
                    body=body_t,
                )
            case ast.VarDecl(typ=typ, name=name, value=value):
                if tail:
                    raise TypecheckError("Missing return")
                value_t = None
                if value:
                    # We check the value before we update the env to handle
                    # things like `int x = x || true`.
                    value_t = tc_expr(env, value)
                    match_type(typ, value_t.type)
                env[name] = typ
                stmt_t = attrs.evolve(
                    stmt,
                    value=value_t
                )
            case ast.Assg(name=name, value=value):
                if tail:
                    raise TypecheckError("Missing return")
                value_t = tc_expr(env, value)
                var_type = get_var(env, name)
                match_type(var_type, value_t.type)
                stmt_t = attrs.evolve(
                    stmt,
                    value=value_t
                )
            case ast.Kurwa():
                if not env[KURWA_VAR]:
                    raise TypecheckError("`kurwa XD` outside of valid if-then branch")
                stmt_t = stmt
            case ast.Chuj():
                if not env[CHUJ_VAR]:
                    raise TypecheckError("`albo chuj` outside of if-else branch")
                stmt_t = stmt
            case _:
                raise ValueError(f"Unsupported statement type: {type(stmt)}")
        block_t.append(stmt_t)

    block_t = ast.Block(stmts=block_t)
    return block_t


def tc_decl(env, decl):
    """
    Typechecks a top-level declaration
    """
    env = env.copy()

    match decl:
        case ast.FunDecl(
                ret=ret,
                args=args,
                body=body,
        ):
            env[RETURN_VAR] = ret
            for (arg_t, arg_n) in args:
                env[arg_n] = arg_t

            enforce_return = ret != t_void
            body_t = tc_block(env, tail=enforce_return, block=body)

            return attrs.evolve(
                decl,
                body=body_t
            )


def tc_program(env, program):
    # First, add all function signatures
    for decl in program.decls:
        match decl:
            case ast.FunDecl(ret=ret, name=name, args=args):
                if name in env:
                    raise TypecheckError(f"Redefinition of function {name}")
                env[name] = ast.TypeFun(
                    ret=ret,
                    args=[arg_t for arg_t, _ in args]
                )
            case _:
                pass

    if 'main' in env and env['main'] != ast.TypeFun(args=[], ret=t_void):
        raise TypecheckError(f"Invalid type of 'main': {env['main']}")

    # Typecheck all declarations in the updated env
    decls_t = []
    for decl in program.decls:
        decl_t = tc_decl(env, decl)
        decls_t.append(decl_t)

    return attrs.evolve(program, decls=decls_t)
