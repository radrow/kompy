"""
Simple AST to JVM bytecode compiler
"""
from . import ast
from . import jvm
from . import typechecker as t


def compile_type(typ):
    """Convert AST type to JVM type descriptor"""
    match typ:
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
            args_desc = ''.join([compile_type(arg) for arg in args])
            return f'({args_desc}){compile_type(ret)}'
    raise ValueError(f"Unknown type: {typ}")


class CompilerEnv:
    """Compilation environment"""
    def __init__(self, class_name):
        self.class_name = class_name
        self.locals = {}  # var_name -> (index, type)
        self.next_local = 0
        self.max_locals = 0
        self.current_block = jvm.INIT_BLOCK
        self.blocks = {jvm.INIT_BLOCK: jvm.Block(name=jvm.INIT_BLOCK)}
        self.label_counter = 0
        self.scope_stack = []

    def push_scope(self):
        """Push a new variable scope (save current locals)"""
        self.scope_stack.append(dict(self.locals))

    def pop_scope(self):
        """Pop variable scope (restore previous locals)"""
        if hasattr(self, 'scope_stack') and self.scope_stack:
            self.locals = self.scope_stack.pop()

    def bind_var(self, name, typ):
        """Bind a variable to a local slot"""
        self.locals[name] = (self.next_local, typ)
        self.next_local += 1
        self.max_locals = max(self.max_locals, self.next_local)

    def get_var(self, name):
        """Get variable's local slot and type"""
        return self.locals[name]

    def emit(self, *instrs):
        """Emit instructions to current block"""
        self.blocks[self.current_block].append(*instrs)

    def new_label(self, hint="", keep=False):
        """Generate a new unique label"""
        self.label_counter += 1 if not keep else 0
        return f"{hint}_{self.label_counter}" if hint else f"L_{self.label_counter}"

    def new_block(self, label):
        """Create a new block with given label"""
        self.blocks[label] = jvm.Block(name=label)
        return label

    def switch_block(self, label):
        """Switch to emitting into a different block"""
        self.current_block = label

    def switch_new_block(self, label):
        """Switch to emitting into a new block"""
        self.new_block(label)
        self.switch_block(label)

    def closed(self, label=None):
        if not label:
            label = self.current_block
        return self.blocks[label].is_closed()


def compile_expr(env, expr):
    """Compile an expression"""
    match expr:
        case ast.Int(v=v):
            env.emit(jvm.Instr.iconst(v))

        case ast.Bool(v=v):
            env.emit(jvm.Instr.iconst(1 if v else 0))

        case ast.String(v=v):
            env.emit(jvm.Instr.ldc(v))

        case ast.Var(name=name):
            index, typ = env.get_var(name)
            if typ in [t.t_int, t.t_bool]:
                env.emit(jvm.Instr.iload(index))
            else:
                env.emit(jvm.Instr.aload(index))

        case ast.Call(fun=fun, args=args):
            # Compile arguments first
            for arg in args:
                compile_expr(env, arg)

            # Handle operators and builtins
            match fun:
                case '+':
                    env.emit(jvm.Instr.iadd())
                case '-':
                    env.emit(jvm.Instr.isub())
                case '*':
                    env.emit(jvm.Instr.imul())
                case '/':
                    env.emit(jvm.Instr.idiv())
                case '%':
                    env.emit(jvm.Instr.irem())
                case '&&':
                    env.emit(jvm.Instr.iand())
                case '||':
                    env.emit(jvm.Instr.ior())
                case '!':
                    env.emit(jvm.Instr.iconst(1), jvm.Instr.ixor())
                case '==' | '!=' | '<' | '>' | '<=' | '>=':
                    # Get the types of the arguments for comparison
                    left_type = args[0].type if len(args) > 0 else None
                    right_type = args[1].type if len(args) > 1 else None
                    compile_comparison(env, fun, left_type, right_type)
                case 'print_int':
                    env.emit(
                        jvm.Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                        jvm.Instr.swap(),
                        jvm.Instr.invokevirtual('java/io/PrintStream/println(I)V')
                    )
                case 'print_bool':
                    env.emit(
                        jvm.Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                        jvm.Instr.swap(),
                        jvm.Instr.invokevirtual('java/io/PrintStream/println(I)V')
                    )
                case 'print_string':
                    env.emit(
                        jvm.Instr.getstatic('java/lang/System/out', 'Ljava/io/PrintStream;'),
                        jvm.Instr.swap(),
                        jvm.Instr.invokevirtual('java/io/PrintStream/println(Ljava/lang/String;)V')
                    )
                case _:
                    # User-defined function
                    typ = ast.TypeFun(args=[arg.type for arg in args], ret=expr.type)
                    env.emit(jvm.Instr.invokestatic(f'{env.class_name}/{fun}{compile_type(typ)}'))
        case _:
            raise ValueError(f"Unhandled expr {expr}")


def compile_comparison(env, op, left_type=None, right_type=None):
    """Compile comparison operators"""
    # Generate unique labels
    true_label = env.new_label("cmp_true")
    label_end = env.new_label("cmp_end")

    # Handle string comparisons differently
    if left_type == t.t_string or right_type == t.t_string:
        if op == '==':
            # For string equality, use String.equals()
            env.emit(jvm.Instr.invokevirtual('java/lang/String/equals(Ljava/lang/Object;)Z'))
            env.emit(jvm.Instr.ifne(true_label))
        elif op == '!=':
            # For string inequality, use String.equals() and negate
            env.emit(jvm.Instr.invokevirtual('java/lang/String/equals(Ljava/lang/Object;)Z'))
            env.emit(jvm.Instr.ifeq(true_label))
        else:
            # Other string comparisons not supported
            raise ValueError(f"String comparison '{op}' not supported")
    else:
        # Choose the right comparison instruction for integers
        cmp_instr = {
            '==': jvm.Instr.if_icmpeq,
            '!=': jvm.Instr.if_icmpne,
            '<': jvm.Instr.if_icmplt,
            '>': jvm.Instr.if_icmpgt,
            '<=': jvm.Instr.if_icmple,
            '>=': jvm.Instr.if_icmpge,
        }[op]

        # Jump to true label if comparison succeeds
        env.emit(cmp_instr(true_label))

    # False case: push 0 and jump to end
    env.emit(jvm.Instr.iconst(0), jvm.Instr.goto(label_end))

    # True case: create block and push 1
    env.switch_new_block(true_label)
    env.emit(jvm.Instr.iconst(1), jvm.Instr.goto(label_end))

    # End: create block for continuation
    env.switch_new_block(label_end)


def compile_stmt(env, stmt):
    """Compile a statement"""
    match stmt:
        case ast.SExpr(expr=expr):
            compile_expr(env, expr)
            if expr.type != t.t_void:
                env.emit(jvm.Instr.pop())

        case ast.VarDecl(typ=typ, name=name, value=value):
            if typ != t.t_void:
                env.bind_var(name, typ)
                if value:
                    compile_expr(env, value)
                    index, _ = env.get_var(name)
                    if typ in [t.t_int, t.t_bool]:
                        env.emit(jvm.Instr.istore(index))
                    else:
                        env.emit(jvm.Instr.astore(index))

        case ast.Assg(name=name, value=value):
            compile_expr(env, value)
            index, typ = env.get_var(name)
            if typ in [t.t_int, t.t_bool]:
                env.emit(jvm.Instr.istore(index))
            else:
                env.emit(jvm.Instr.astore(index))

        case ast.Return(expr=expr):
            if expr:
                compile_expr(env, expr)
                if expr.type in [t.t_int, t.t_bool]:
                    env.emit(jvm.Instr.ireturn())
                else:
                    env.emit(jvm.Instr.areturn())
            else:
                env.emit(jvm.Instr.return_())

        case ast.If(cond=cond, then_block=then_block, else_block=else_block):
            compile_if(env, cond, then_block, else_block)

        case ast.While(cond=cond, body=body):
            compile_while(env, cond, body)

        case _:
            raise ValueError(f"Unhandled stmt {stmt}")


def compile_cond(env, cond, label_then):
    """Compile a branching flow"""

    # Handle condition - check if it's a comparison that we can optimize
    if isinstance(cond, ast.Call) and cond.fun in ['==', '!=', '<', '>', '<=', '>=']:
        # Direct comparison - compile operands and use conditional jump
        compile_expr(env, cond.args[0])
        compile_expr(env, cond.args[1])

        # Check if we're comparing strings
        left_type = cond.args[0].type if len(cond.args) > 0 else None
        right_type = cond.args[1].type if len(cond.args) > 1 else None

        if left_type == t.t_string or right_type == t.t_string:
            # Handle string comparisons
            if cond.fun == '==':
                env.emit(jvm.Instr.invokevirtual('java/lang/String/equals(Ljava/lang/Object;)Z'))
                env.emit(jvm.Instr.ifne(label_then))
            elif cond.fun == '!=':
                env.emit(jvm.Instr.invokevirtual('java/lang/String/equals(Ljava/lang/Object;)Z'))
                env.emit(jvm.Instr.ifeq(label_then))
            else:
                raise ValueError(f"String comparison '{cond.fun}' not supported")
        else:
            # Integer comparisons
            cmp_instr = {
                '==': jvm.Instr.if_icmpeq,
                '!=': jvm.Instr.if_icmpne,
                '<': jvm.Instr.if_icmplt,
                '>': jvm.Instr.if_icmpgt,
                '<=': jvm.Instr.if_icmple,
                '>=': jvm.Instr.if_icmpge,
            }[cond.fun]

            env.emit(cmp_instr(label_then))
    else:
        # General condition - compile and test for non-zero
        compile_expr(env, cond)
        env.emit(jvm.Instr.ifne(label_then))


def compile_if(env, cond, then_block, else_block):
    """Compile if statement"""
    # Generate unique labels
    label_then = env.new_label("if_then")
    label_end = env.new_label("if_end", keep=True)

    # Compile the conditional jump
    compile_cond(env, cond, label_then)

    # Else block (if exists) or fall through
    if else_block:
        compile_block(env, else_block)

    needs_end = False

    # Only add goto to end if the current block isn't closed
    if env.closed():
        needs_end = True
        env.emit(jvm.Instr.goto(label_end))

    # Then block
    env.switch_new_block(label_then)
    compile_block(env, then_block)

    # Only add goto to end if the then block isn't closed
    if not env.closed():
        needs_end = True
        env.emit(jvm.Instr.goto(label_end))

    # Only create and switch to end block if at least one branch needs it
    if needs_end:
        env.switch_new_block(label_end)


def compile_while(env, cond, body):
    label_cond = env.new_label('while_cond')
    label_body = env.new_label('while_body', keep=True)
    label_end = env.new_label("while_end", keep=True)

    # Jump to the condition block
    env.emit(jvm.Instr.goto(label_cond))

    # Compile condition
    env.switch_new_block(label_cond)
    compile_cond(env, cond, label_body)
    env.emit(jvm.Instr.goto(label_end))

    # Compile body
    env.switch_new_block(label_body)
    compile_block(env, body)
    if not env.closed():
        env.emit(jvm.Instr.goto(label_cond))

    env.switch_new_block(label_end)


def compile_block(env, block):
    """Compile a block of statements"""
    # Push a new scope for this block
    env.push_scope()

    # Compile all statements
    for stmt in block.stmts:
        compile_stmt(env, stmt)

    # Pop the scope when exiting the block
    env.pop_scope()


def compile_function(class_name, fdecl):
    """Compile a function declaration"""
    env = CompilerEnv(class_name)

    # Bind parameters
    for arg_type, arg_name in fdecl.args:
        env.bind_var(arg_name, arg_type)

    # Compile body
    compile_block(env, fdecl.body)

    # Ensure void functions return (only if the current block isn't already closed)
    if fdecl.ret == t.t_void and not env.blocks[env.current_block].is_closed():
        env.emit(jvm.Instr.return_())

    return jvm.Method(
        visibility='public',
        name=fdecl.name,
        static=True,
        args=[compile_type(arg_type) for arg_type, _ in fdecl.args],
        ret=compile_type(fdecl.ret),
        stack=1000,  # Conservative estimate
        local=env.max_locals,
        blocks=env.blocks
    )


def compile_program(program):
    """Compile a program to JVM class"""
    methods = []

    for decl in program.decls:
        if isinstance(decl, ast.FunDecl):
            methods.append(compile_function(program.name, decl))

    return jvm.Class(
        name=program.name,
        visibility='public',
        superclass='java/lang/Object',
        methods=methods
    )
