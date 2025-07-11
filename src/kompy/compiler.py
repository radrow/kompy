import typing
import attr

from . import ast
from . import jvm
from . import typecheck as t
from .jvm import Instr, Block


@attr.s(auto_attribs=True)
class Feed:
    """
    Helper class to provide label names.
    """
    feed: int = 0

    def get(self, name):
        """
        Returns the most recent label.
        """
        return f"{name}__{self.feed}"

    def next(self, name):
        """
        Creates a fresh label and returns it.
        """
        self.feed += 1
        return self.get(name)


@attr.s(frozen=True, auto_attribs=True)
class Env:
    """
    Local context
    """
    label_base: str
    local: typing.Dict[str, int] = attr.Factory(dict)
    current_block: str = jvm.INIT_BLOCK
    next_block: typing.Optional[str] = None
    blocks: typing.Dict[str, Block] = attr.Factory(dict)
    feed: Feed = attr.ib(factory=Feed)

    def __attrs_post_init__(self):
        self.blocks[jvm.INIT_BLOCK] = Block()

    def append(self, instr: typing.Union[Instr, typing.Iterable[Instr]]):
        """
        Appends an instruction or a collection of instructions to the current
        block.
        """
        self.blocks[self.current_block].append(instr)

    def next_label(self):
        """
        Creates and returns a fresh label
        """
        return self.feed.next(self.label_base)

    def new_block(self):
        """
        Creates a new block under a fresh label.
        """
        label = self.next_label()
        self.blocks[label] = Block()
        return label

    def in_block(self, label):
        """
        Switches the block under focus.
        """
        return attr.evolve(self, current_block=label)

    def with_next(self, label):
        """
        Creates an env with different continuation.
        """
        return attr.evolve(self, next_block=label)

    def close_block(self):
        """
        Ends the current block. If continuation exists, appends a jump.
        """
        if self.next_block:
            self.append(Instr.goto(self.next_block))


def compile_fun(env, fun) -> Env:
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
        case '==':
            # For comparisons, we need to handle them differently
            # This is a simplified approach - in a real compiler you'd want
            # to use proper conditional jumps
            env.append(Instr.isub())  # Subtract to get 0 if equal
            # Convert 0 to 1 (true) and non-zero to 0 (false)
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.ifeq(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
        case '!=':
            env.append(Instr.isub())
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.ifne(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
        case '<':
            env.append(Instr.isub())
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.iflt(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
        case '>':
            env.append(Instr.isub())
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.ifgt(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
        case '<=':
            env.append(Instr.isub())
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.ifle(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
        case '>=':
            env.append(Instr.isub())
            label_true = env.next_label()
            label_false = env.next_label()
            label_end = env.next_label()
            env.append(Instr.ifge(label_true))
            env.append(Instr.goto(label_false))
            env.blocks[label_true] = Block()
            env.blocks[label_true].append(Instr.iconst(1))
            env.blocks[label_true].append(Instr.goto(label_end))
            env.blocks[label_false] = Block()
            env.blocks[label_false].append(Instr.iconst(0))
            env.blocks[label_false].append(Instr.goto(label_end))
            env.blocks[label_end] = Block()
            env = env.in_block(label_end)
    
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
        case ast.Var(name=name) if expr.type in [t.t_int, t.t_bool]:
            env.append(Instr.iload(env.local[name]))
        case ast.Var(name=name):
            env.append(Instr.aload(env.local[name]))
        case ast.Call(fun=fun, args=args):
            for arg in args:
                env = compile_expr(env, arg)
            env = compile_fun(env, fun)
    return env


def compile_cond(env: Env, cond: ast.Expr) -> typing.Tuple[Env, Env, Env]:
    """
    Compiles an `if` expression/statement. Returns envs: then, else, after
    """
    label_after = env.new_block()

    label_then = env.new_block()
    label_else = env.new_block()

    env = compile_expr(env, cond)

    env.append(Instr.ifne(label_then))
    env.append(Instr.goto(label_else))

    return (
        env.in_block(label_then).with_next(label_after),
        env.in_block(label_else).with_next(label_after),
        env.in_block(label_after),
    )


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
                if expr:
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
                env_then, env_else, env_after = compile_cond(env, cond)
                compile_block(env_then, then_block)
                compile_block(env_else, else_block)

                # We continue in the updated env
                env = env_after

    env.close_block()
