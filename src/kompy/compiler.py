"""
AST lowering to Control Flow Graph
"""
import typing
import attrs
from attrs import frozen, mutable, field

from . import ast
from . import jvm
from . import typechecker as t
from .jvm import Instr, Block


@mutable(auto_attribs=True)
class Feed:
    """
    Helper class to provide unique labels.
    """
    feed: int = 0

    def get(self, name, hint=None):
        """
        Returns the most recent label.
        """
        hint_str = f'__{hint}' if hint else ''
        return f"{name}{hint_str}__{self.feed}"

    def next(self, name, hint=None):
        """
        Creates a fresh label and returns it.
        """
        self.feed += 1
        return self.get(name, hint=hint)


@frozen(auto_attribs=True)
class GlobEnv:
    """
    Global context
    """
    current_class: str


@frozen(auto_attribs=True)
class Env:
    """
    Local context
    """
    global_env: GlobEnv
    current_method: str
    local: typing.Dict[str, int] = field(factory=dict)
    next_local: int = 0
    current_block: str = jvm.INIT_BLOCK
    next_block: typing.Optional[str] = None
    blocks: typing.Dict[str, Block] = field(factory=dict)
    feed: Feed = field(factory=Feed)

    def __attrs_post_init__(self):
        if jvm.INIT_BLOCK not in self.blocks:
            self.blocks[jvm.INIT_BLOCK] = Block()

    def bind_var(self, name):
        local = self.local.copy()
        local[name] = self.next_local
        return attrs.evolve(self, local=local, next_local=self.next_local+1)

    def get_var(self, name):
        return self.local[name]

    def append(self, instr: typing.Union[Instr, typing.Iterable[Instr]]):
        """
        Appends an instruction or a collection of instructions to the current
        block.
        """
        self.blocks[self.current_block].append(instr)

    def next_label(self, hint=None):
        """
        Creates and returns a fresh label
        """
        return self.feed.next(self.current_method, hint)

    def new_block(self, hint=None):
        """
        Creates a new block under a fresh label.
        """
        label = self.next_label(hint=hint)
        self.blocks[label] = Block()
        return label

    def in_block(self, label):
        """
        Switches the block under focus.
        """
        env = attrs.evolve(self, current_block=label)
        return env

    def with_next(self, label):
        """
        Creates an env with different continuation.
        """
        return attrs.evolve(self, next_block=label)

    def close_block(self):
        """
        Ends the current block. If continuation exists, appends a jump.
        """
        if self.next_block:
            self.append(Instr.goto(self.next_block))


def compile_fun(env, fun, typ) -> Env:
    """
    Compiles a function. Assumes that args have already been placed where
    they have to be.
    """
    # Handle operators
    match fun:
        case '+':
            env.append(Instr.iadd())
        case '-':
            env.append(Instr.isub())
        case '*':
            env.append(Instr.imul())
        case '/':
            env.append(Instr.idiv())
        case '%':
            env.append(Instr.irem())
        case '&&':
            env.append(Instr.iand())
        case '||':
            env.append(Instr.ior())
        case '!':
            env.append([
                Instr.iconst(1),
                Instr.ixor(),
            ])
        case '==' | '!=' | '<' | '>' | '<=' | '>=':
            env = compile_cmp(env, fun)
        # Handle builtins
        case 'print_int':
            env.append([
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(I)V'),
            ])
        case 'print_bool':
            env.append([
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(I)V'),
            ])
        case 'print_string':
            env.append([
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(Ljava/lang/String;)V'),
            ])
        # Handle user-defined functions
        case _:
            # Assume it's a user-defined function call
            # Use the current_method from the environment to get the class name
            class_name = env.global_env.current_class
            env.append(Instr.invokestatic(f'{class_name}/{fun}{compile_type(typ)}'))

    return env


def compile_cmp(env: Env, op: str) -> Env:
    """
    Compiles a comparison operator. Generates code that leaves 1 (true) or 0
    (false) on the stack.
    """
    # We subtract one operand from another, since the comparison is against 0
    env.append(Instr.isub())

    # Choose the appropriate conditional jump instruction
    jump_instr = {
        '==': Instr.ifeq,
        '!=': Instr.ifne,
        '<': Instr.iflt,
        '>': Instr.ifgt,
        '<=': Instr.ifle,
        '>=': Instr.ifge,
    }[op]

    op_hint = {
        '==': 'eq',
        '!=': 'ne',
        '<': 'lt',
        '>': 'gt',
        '<=': 'le',
        '>=': 'ge',
    }[op]

    def compile_then_branch(env):
        env.append(Instr.iconst(1))
        return env

    def compile_else_branch(env):
        env.append(Instr.iconst(0))
        return env

    env = compile_cond(
        env,
        compile_then=compile_then_branch,
        compile_else=compile_else_branch,
        decider=jump_instr,
        hint=op_hint
    )

    return env


def compile_expr(env: Env, expr) -> Env:
    """
    Compiles an expression. Returns updated env.
    """
    match expr:
        case ast.Int(v=v):
            env.append(Instr.iconst(v))
        case ast.Bool(v=v):
            env.append(Instr.iconst(int(v)))
        case ast.String(v=v):
            env.append(Instr.ldc(v))
        case ast.Var(name=name) if expr.type in [t.t_int, t.t_bool]:
            env.append(Instr.iload(env.local[name]))
        case ast.Var(name=name):
            env.append(Instr.aload(env.local[name]))
        case ast.Call(fun=fun, args=args):
            typ = ast.TypeFun(args=[arg.type for arg in args], ret=expr.type)

            for arg in args:
                env = compile_expr(env, arg)
            env = compile_fun(env, fun, typ)
    return env


def compile_cond(env: Env, compile_then, compile_else, decider=Instr.ifne, hint=None) -> Env:
    """
    Compiles an `if` expression/statement. Returns the environment after the conditional.
    """
    label_then = env.new_block(hint=f'{hint}_true' if hint else None)
    label_else = env.new_block(hint=f'{hint}_false' if hint else None)
    label_after = env.new_block(hint=f'{hint}_after' if hint else None)

    env.append(decider(label_then))
    env.append(Instr.goto(label_else))

    env_then = compile_then(env.in_block(label_then).with_next(label_after))
    env_then.close_block()

    env_else = compile_else(env.in_block(label_else).with_next(label_after))
    env_else.close_block()

    # Since blocks are shared, we just need to return the after environment
    return env.in_block(label_after)


def compile_block(env: Env, block: ast.Block):
    """
    Compiles a block of statements.
    """
    for stmt in block.stmts:
        match stmt:
            case ast.SExpr(expr=expr):
                env = compile_expr(env, expr)
                if expr.type != t.t_void:
                    # Discard the result
                    env.append(Instr.pop())

            case ast.Return(expr=expr):
                if expr is not None:
                    env = compile_expr(env, expr)
                    if expr.type in [t.t_int, t.t_bool]:
                        env.append(Instr.ireturn())
                    else:
                        env.append(Instr.areturn())
                else:
                    env.append(Instr.return_())

            case ast.If(cond=cond,
                        then_block=then_block,
                        else_block=else_block,
                        ):
                # Replace no `else` with `{ }`
                if not else_block:
                    else_block = ast.Block(stmts=[])

                # Compile condition
                env = compile_expr(env, cond)

                # Compile branches. Doing it this way because Python is stupid af
                def compile_then(env, e=then_block):
                    return compile_block(env, e)

                def compile_else(env, e=else_block):
                    return compile_block(env, e)

                env = compile_cond(
                    env,
                    compile_then=compile_then,
                    compile_else=compile_else,
                    hint='if'
                )
            case ast.VarDecl(typ=typ, name=name, value=value):
                if typ != t.t_void:
                    env = env.bind_var(name)
                if value:
                    env = compile_expr(env, value)
                    match typ:
                        case ast.TypeVar(name='int'):
                            store = Instr.istore
                        case ast.TypeVar(name='bool'):
                            store = Instr.istore
                        case ast.TypeVar(name='void'):
                            store = None
                        case _:
                            store = Instr.astore
                    if store:
                        loc = env.get_var(name)
                        env.append(store(loc))
            case ast.Assg(name=name, value=value):
                env = compile_expr(env, value)
                match value.type:
                    case ast.TypeVar(name='int'):
                        store = Instr.istore
                    case ast.TypeVar(name='bool'):
                        store = Instr.istore
                    case ast.TypeVar(name='void'):
                        store = None
                    case _:
                        store = Instr.astore
                if store:
                    loc = env.get_var(name)
                    env.append(store(loc))
    env.close_block()
    return env


def compile_type(ty: ast.Type) -> str:
    """
    Represents a type in terms of JVM
    """
    match ty:
        case ast.TypeVar(name='int'):
            return 'I'
        case ast.TypeVar(name='bool'):
            return 'I'
        case ast.TypeVar(name='string'):
            return 'Ljava/lang/String;'
        case ast.TypeVar(name='void'):
            return 'V'
        case ast.TypeArr(el=el):
            return '[' + compile_type(el)
        case ast.TypeFun(args=args, ret=ret):
            args_t = ''.join([compile_type(arg) for arg in args])
            return f'({args_t}){compile_type(ret)}'
    raise ValueError(f"Can't recognize {ty} as a JVM type")


def compile_function(env: GlobEnv, fdecl: ast.FunDecl) -> jvm.Method:
    env = Env(global_env=env, current_method=fdecl.name)

    for (_arg_t, arg_name) in fdecl.args:
        env = env.bind_var(arg_name)
    env = compile_block(env, fdecl.body)
    
    # Add return instruction for void functions if not already present
    if fdecl.ret.name == 'void':
        env.append(Instr.return_())

    return jvm.Method(
        visibility='public',
        name=fdecl.name,
        static=True,
        args=[compile_type(arg_t) for (arg_t, _) in fdecl.args],
        ret=compile_type(fdecl.ret),
        stack=1000,  # TODO XD
        local=env.next_local,
        blocks=env.blocks,
    )


def compile_program(program: ast.Program) -> jvm.Class:
    env = GlobEnv(current_class=program.name)

    methods = []
    for f in program.decls:
        match f:
            case ast.FunDecl():
                methods.append(compile_function(env, f))

    return jvm.Class(
        name=program.name,
        visibility='public',
        superclass='java/lang/Object',
        methods=methods,
    )
