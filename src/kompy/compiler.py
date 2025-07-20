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
        hint_str = f'${hint}' if hint else ''
        return f"{name}{hint_str}$__{self.feed}"

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
    next_block: typing.Union[None, str, typing.Tuple[str, str]] = None
    blocks: typing.Dict[str, Block] = field(factory=dict)
    feed: Feed = field(factory=Feed)

    def __attrs_post_init__(self):
        if jvm.INIT_BLOCK not in self.blocks:
            self.blocks[jvm.INIT_BLOCK] = Block()

    def bind_var(self, name, typ):
        local = self.local.copy()
        local[name] = (self.next_local, typ)
        return attrs.evolve(self, local=local, next_local=self.next_local+1)

    def get_var(self, name):
        return self.local[name]

    def append(self, *instrs: Instr):
        """
        Appends an instruction or a collection of instructions to the current
        block.
        """
        self.blocks[self.current_block].append(*instrs)

    def next_label(self, hint=None, next_feed=True):
        """
        Creates and returns a fresh label
        """
        feeder = self.feed.next if next_feed else self.feed.get
        return feeder(self.current_method, hint=hint)

    def new_block(self, hint=None, next_feed=True):
        """
        Creates a new block under a fresh label.
        """
        label = self.next_label(hint=hint, next_feed=next_feed)
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
        match self.next_block:
            case str():
                # Jump to the next block
                self.append(Instr.goto(self.next_block))
            case (next_then, next_else):
                # Branch between blocks
                self.append(
                    Instr.ifne(next_then),
                    Instr.goto(next_else),
                )
            case None:
                pass


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
            env.append(
                Instr.iconst(1),
                Instr.ixor(),
            )
        case '==' | '!=' | '<' | '>' | '<=' | '>=':
            env = compile_cmp(env, fun)
        # Handle builtins
        case 'print_int':
            env.append(
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(I)V'),
            )
        case 'print_bool':
            env.append(
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(I)V'),
            )
        case 'print_string':
            env.append(
                Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                Instr.swap(),
                Instr.invokevirtual('java/io/PrintStream/println(Ljava/lang/String;)V'),
            )
        # Handle user-defined functions
        case _:
            # Assume it's a user-defined function call
            # Use the current_method from the environment to get the class name
            class_name = env.global_env.current_class
            env.append(Instr.invokestatic(f'{class_name}/{fun}{compile_type(typ)}'))

    return env


def op_hint(op):
    return {
        '==': 'eq',
        '!=': 'ne',
        '<': 'lt',
        '>': 'gt',
        '<=': 'le',
        '>=': 'ge',
    }[op]


def zcmp_instr(op):
    return {
        '==': Instr.ifeq,
        '!=': Instr.ifne,
        '<': Instr.iflt,
        '>': Instr.ifgt,
        '<=': Instr.ifle,
        '>=': Instr.ifge,
    }[op]


def icmp_instr(op):
    return {
        '==': Instr.if_icmpeq,
        '!=': Instr.if_icmpne,
        '<': Instr.if_icmplt,
        '>': Instr.if_icmpgt,
        '<=': Instr.if_icmple,
        '>=': Instr.if_icmpge,
    }[op]


def compile_cmp(env: Env, op: str) -> Env:
    """
    Compiles a comparison operator. If there is a branch, applies the
    appropriate conditional jump. Otherwise generates code that leaves 1 (true)
    or 0 (false) on the stack.
    """

    match env.next_block:
        case (if_then, if_else):
            # Control flow branches depending on the result
            label_then, label_else = if_then, if_else

            # We consume the branching continuation. No need to keep track of
            # it after.
            env_after = env.with_next(None)
        case _:
            # The result is to be pushed onto the stack (as `1` or `0`). We
            # need to generate blocks for those pushes.
            hint = op_hint(op)
            label_then = env.new_block(hint=f'{hint}_true')
            label_else = env.new_block(hint=f'{hint}_false', next_feed=False)

            # Inherit the continuation if defined, otherwise start a new block.
            if env.next_block:
                label_after = env.next_block
            else:
                label_after = env.new_block(hint=f'{hint}_after', next_feed=False)

            # Block pushing 1
            env_then = env.in_block(label_then).with_next(label_after)
            env_then.append(Instr.iconst(1))
            env_then.close_block()

            # Block pushing 0
            env_else = env.in_block(label_else).with_next(label_after)
            env_else.append(Instr.iconst(0))
            env_else.close_block()

            # Continue
            env_after = env.in_block(label_after)

    # Apply branching
    env.append(
        icmp_instr(op)(label_then),
        Instr.goto(label_else),
    )

    return env_after


def compile_expr(env: Env, expr) -> Env:
    """
    Compiles an expression. Returns updated env.
    """
    match expr:
        case ast.Int(v=v):
            # Just push the int
            env.append(Instr.iconst(v))

        case ast.Bool(v=v):
            # Bools are ints
            env.append(Instr.iconst(int(v)))

        case ast.String(v=v):
            # Push the string
            env.append(Instr.ldc(v))

        case ast.Var(name=name):
            # Resolve the location of the variable
            (loc, _typ) = env.get_var(name)

            # Load from the storage
            if expr.type in [t.t_int, t.t_bool]:
                load = Instr.iload
            else:
                load = Instr.aload

            env.append(load(loc))

        case ast.Call(fun=fun, args=args):
            # Reconstruct the type of the function
            typ = ast.TypeFun(args=[arg.type for arg in args], ret=expr.type)

            # Backup the continuation
            next_block = env.next_block

            # We want the args to be placed on the stack. We must temporarily
            # forget about the continuation.
            env = env.with_next(None)
            for arg in args:
                env = compile_expr(env, arg)

            # Bring back the original continuation and compile the function call.
            env = env.with_next(next_block)
            env = compile_fun(env, fun, typ)
    return env


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

            case ast.If(
                    cond=cond,
                    then_block=then_block,
                    else_block=else_block,
            ):
                # Replace no `else` with `{ }`.
                if not else_block:
                    else_block = ast.Block(stmts=[])

                # Add blocks for both branches.
                label_then = env.new_block(hint='if_true')
                label_else = env.new_block(hint='if_false', next_feed=False)

                # If needed, add a continuation block if there is anything
                # after the `if`.
                if env.next_block:
                    label_after = env.new_block(hint='if_after', next_feed=False)
                else:
                    label_after = None
                env_after = env.in_block(label_after)

                # Compile the condition with branching continuation.
                env = env.with_next((label_then, label_else))
                env = compile_expr(env, cond)
                env.close_block()

                # Compile the positive and negative branch. Inherit the
                # continuation.
                env_then = env.in_block(label_then).with_next(label_after)
                env_then = compile_block(env_then, then_block)

                env_else = env.in_block(label_else).with_next(label_after)
                env_else = compile_block(env_else, else_block)

                # We move to the continuation.
                env = env_after

            case ast.VarDecl(typ=typ, name=name, value=value):
                # Bind the variable in the env if it's not void
                if typ != t.t_void:
                    env = env.bind_var(name, typ)

                # Assign the initial value if provided
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
                        (loc, _typ) = env.get_var(name)
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
                    (loc, _typ) = env.get_var(name)
                    env.append(store(loc))

    # We apply any remaining continuation and return the updated env
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
            return 'Z'
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

    for (arg_t, arg_name) in fdecl.args:
        env = env.bind_var(arg_name, arg_t)

    # Make sure void functions have a return
    if fdecl.ret == t.t_void:
        void_return = env.new_block(hint='void_ret')
        env.in_block(void_return).append(Instr.return_())
        env = env.with_next(void_return)

    env = compile_block(env, fdecl.body)

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
