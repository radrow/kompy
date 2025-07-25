"""
RISC-V compiler backend - lowers typed AST to RISC-V RV32IM assembly
Inspired by RISCVCodegen.fs
"""
import typing
import json
from dataclasses import dataclass, field

from . import ast
from . import riscv
from .riscv import Instr, Reg


@dataclass
class Storage:
    """Storage information for variables"""
    pass


@dataclass
class RegStorage(Storage):
    """Variable stored in integer register"""
    reg: str


@dataclass
class LabelStorage(Storage):
    """Variable stored in memory at a label"""
    label: str


@dataclass
class CompilerEnv:
    """RISC-V compilation environment"""
    program: riscv.Program
    current_function: typing.Optional[riscv.Function] = None
    current_block: typing.Optional[riscv.Block] = None

    # Target register for expression results
    target: int = 0

    # Variable storage mapping
    var_storage_stack: typing.List[typing.Dict[str, Storage]] = field(default_factory=lambda: [{}])

    # Label generation
    label_counter: int = 0
    string_counter: int = 0

    # Register allocation
    var_counter: int = 0  # Global counter for variable register allocation

    @property
    def var_storage(self) -> typing.Dict[str, Storage]:
        """Get current variable storage (top of scope stack)"""
        return self.var_storage_stack[-1]

    def push_scope(self):
        """Push a new scope for variable bindings"""
        # Copy current scope to preserve outer bindings
        current_scope = self.var_storage_stack[-1].copy()
        self.var_storage_stack.append(current_scope)

    def pop_scope(self):
        """Pop the current scope, restoring previous variable bindings"""
        if len(self.var_storage_stack) > 1:
            self.var_storage_stack.pop()
        else:
            raise ValueError("Cannot pop the global scope")

    def add_variable(self, name: str, storage: Storage):
        """Add a variable to the current scope"""
        self.var_storage_stack[-1][name] = storage

    def lookup_variable(self, name: str) -> typing.Optional[Storage]:
        """Look up a variable in the scope stack (innermost first)"""
        # Search from innermost to outermost scope
        for scope in reversed(self.var_storage_stack):
            if name in scope:
                return scope[name]
        return None

    def allocate_var_register(self) -> str:
        """Allocate a new saved register for a variable"""
        reg = f"s{self.var_counter}"
        self.var_counter += 1
        return reg

    def new_label(self, hint: str = "label") -> str:
        """Generate a new unique label"""
        label = f"{hint}_{self.label_counter}"
        self.label_counter += 1
        return label

    def new_string(self) -> str:
        """Generate a new string label"""
        label = f"string_{self.string_counter}"
        self.string_counter += 1
        return label

    def get_target_reg(self) -> str:
        """Get target register for integer values"""
        return f"t{self.target}"

    def emit(self, *instrs: Instr):
        """Emit one or more instructions to current block"""
        if not self.current_block:
            raise ValueError("No current block to emit to")

        # Don't emit to a closed block
        if self.current_block.is_closed():
            return

        for instr in instrs:
            self.current_block.append(instr)

    def start_function(self, name: str, args: typing.List[typing.Tuple[ast.Type, str]], is_main: bool = False):
        """Start a new function"""
        self.current_function = riscv.Function(name=name, global_=is_main)
        self.current_block = riscv.Block(name=riscv.INIT_BLOCK)
        self.current_function.blocks[riscv.INIT_BLOCK] = self.current_block

        # Reset register allocation for function
        self.target = 0
        self.var_counter = 0

        # Reset scope stack to just the global scope for function parameters
        self.var_storage_stack = [{}]

        # Set up function arguments in registers
        if len(args) >= 8:
            raise ValueError("Can't handle that many arguments!")
        for i, (_typ, arg_name) in enumerate(args):
            if i < 8:  # First 8 args go in a0-a7
                arg_reg = f"a{i}"
                # Copy argument to a saved register to preserve it across function calls
                saved_reg = self.allocate_var_register()
                self.emit(Instr.mv(saved_reg, arg_reg).with_comment(f"Save parameter '{arg_name}'"))
                self.add_variable(arg_name, RegStorage(saved_reg))

    def end_function(self):
        """Finish current function"""
        if self.current_function:
            self.program.add_function(self.current_function)
        self.current_function = None
        self.current_block = None

    def add_data(self, directive: str):
        """Add data directive to program"""
        self.program.add_data(directive)


def is_int_type(typ: ast.Type) -> bool:
    """Check if type is integer-like"""
    return isinstance(typ, ast.TypeVar) and typ.name in ['int', 'bool']


def is_string_type(typ: ast.Type) -> bool:
    """Check if type is string"""
    return isinstance(typ, ast.TypeVar) and typ.name == 'string'


def is_void_type(typ: ast.Type) -> bool:
    """Check if type is void"""
    return isinstance(typ, ast.TypeVar) and typ.name == 'void'


def compile_expr(env: CompilerEnv, expr: ast.Expr):
    """
    Compile an expression and store the result in the target register specified by env.
    """
    match expr:
        case ast.Int(v=value):
            target_reg = env.get_target_reg()
            env.emit(
                Instr.li(target_reg, value)
                .with_comment(f"Load integer {value}")
            )

        case ast.Bool(v=value):
            target_reg = env.get_target_reg()
            int_value = 1 if value else 0
            env.emit(
                Instr.li(target_reg, int_value)
                .with_comment(f"Load boolean {value}")
            )

        case ast.String(v=value):
            target_reg = env.get_target_reg()
            string_label = env.new_string()
            env.add_data(f'{string_label}: .string {json.dumps(value)}')
            env.emit(
                Instr.la(target_reg, string_label)
            )

        case ast.Var(name=name):
            compile_var_read(env, name, expr.type)

        case ast.Call(fun=fun, args=args):
            compile_call(env, fun, args)

        case _:
            raise ValueError(f"Unsupported expression type: {type(expr)}")


def compile_arithmetic_op(env: CompilerEnv, op: str, args: typing.List[ast.Expr]):
    """Compile arithmetic operators (+, -, *, /, %)"""
    if len(args) != 2:
        raise ValueError(f"Binary operator {op} requires exactly 2 arguments")

    # Compile left operand to target register
    left_target = env.target
    compile_expr(env, args[0])
    left_reg = env.get_target_reg()

    # Save left operand to a temporary saved register to preserve it across right operand compilation
    temp_reg = env.allocate_var_register()
    env.emit(Instr.mv(temp_reg, left_reg).with_comment("Save left operand"))

    # Compile right operand to next register
    env.target += 1
    compile_expr(env, args[1])
    right_reg = env.get_target_reg()

    # Reset target back to left operand register for result
    env.target = left_target
    target_reg = env.get_target_reg()

    # Integer arithmetic - use saved left operand
    match op:
        case '+':
            env.emit(Instr.add(target_reg, temp_reg, right_reg).with_comment("Integer addition"))
        case '-':
            env.emit(Instr.sub(target_reg, temp_reg, right_reg).with_comment("Integer subtraction"))
        case '*':
            env.emit(Instr.mul(target_reg, temp_reg, right_reg).with_comment("Integer multiplication"))
        case '/':
            env.emit(Instr.div(target_reg, temp_reg, right_reg).with_comment("Integer division"))
        case '%':
            env.emit(Instr.rem(target_reg, temp_reg, right_reg).with_comment("Integer remainder"))


def compile_logical_op(env: CompilerEnv, op: str, args: typing.List[ast.Expr]):
    """Compile logical operators (&&, ||)"""
    if len(args) != 2:
        raise ValueError(f"Logical operator {op} requires exactly 2 arguments")

    left_target = env.target
    compile_expr(env, args[0])
    left_reg = env.get_target_reg()

    env.target += 1
    compile_expr(env, args[1])
    right_reg = env.get_target_reg()

    env.target = left_target
    target_reg = env.get_target_reg()

    match op:
        case '&&':
            env.emit(Instr.and_(target_reg, left_reg, right_reg).with_comment("Logical AND"))
        case '||':
            env.emit(Instr.or_(target_reg, left_reg, right_reg).with_comment("Logical OR"))


def compile_unary_op(env: CompilerEnv, op: str, args: typing.List[ast.Expr]):
    """Compile unary operators (!)"""
    if op == '!':
        if len(args) != 1:
            raise ValueError("Unary ! operator requires exactly 1 argument")

        compile_expr(env, args[0])
        arg_reg = env.get_target_reg()
        target_reg = env.get_target_reg()
        env.emit(Instr.seqz(target_reg, arg_reg).with_comment("Logical NOT"))
    else:
        raise ValueError(f"Unsupported unary operator: {op}")


def compile_builtin_function(env: CompilerEnv, fun: str, args: typing.List[ast.Expr]):
    """Compile built-in functions (print_int, print_bool, print_string)"""
    match fun:
        case 'print_int':
            compile_expr(env, args[0])
            arg_reg = env.get_target_reg()
            env.emit(
                Instr.mv(Reg.A0, arg_reg)
                .with_comment("Print integer"),
                Instr.li(Reg.A7, riscv.SysCall.PRINT_INT),
                Instr.ecall(),
            )

        case 'print_bool':
            compile_expr(env, args[0])
            arg_reg = env.get_target_reg()
            env.emit(
                Instr.mv(Reg.A0, arg_reg)
                .with_comment("Print boolean"),

                Instr.li(Reg.A7, riscv.SysCall.PRINT_INT),
                Instr.ecall(),
            )

        case 'print_string':
            compile_expr(env, args[0])
            arg_reg = env.get_target_reg()
            env.emit(
                Instr.mv(Reg.A0, arg_reg)
                .with_comment("Print string"),
                Instr.li(Reg.A7, riscv.SysCall.PRINT_STRING),
                Instr.ecall(),
            )

        case _:
            raise ValueError(f"Unsupported built-in function: {fun}")


def compile_function_call(env: CompilerEnv, fun: str, args: typing.List[ast.Expr]):
    """Compile user-defined function call"""

    # Compile arguments and load into temporary registers
    old_target = env.target
    for i, arg in enumerate(args, old_target):
        if i < Reg.T_REGS:
            env.target = i  # Use register index as target
            compile_expr(env, arg)
        else:
            raise ValueError("Too many args :(")
    env.target = old_target

    # Save temporary and argument registers
    saved_regs = [reg for reg in riscv.CALLER_SAVED if reg != env.get_target_reg()]
    save_registers(env, saved_regs)

    # Load arguments to argument registers
    for ia, (it, arg) in enumerate(enumerate(args, old_target)):
        tmp_reg = Reg.T(it)
        arg_reg = Reg.A(ia)
        env.emit(
                Instr.mv(arg_reg, tmp_reg)
                .with_comment(f"Load argument {ia}")
        )

    # Function call
    env.emit(Instr.jal(Reg.RA, fun).with_comment(f"Call function {fun}"))

    # Move return value to target register
    target_reg = env.get_target_reg()
    env.emit(Instr.mv(target_reg, Reg.A0).with_comment("Move return value"))

    # Restore saved registers
    restore_registers(env, saved_regs)


def compile_var_read(env: CompilerEnv, name: str, var_type: ast.Type):
    """Compile variable access and load into target register"""
    storage = env.lookup_variable(name)
    if storage is None:
        raise ValueError(f"Variable '{name}' not found in storage")

    if is_void_type(var_type):
        # Unit-typed variable is ignored
        pass
    else:
        # Integer-like values
        target_reg = env.get_target_reg()
        match storage:
            case RegStorage(reg=reg):
                env.emit(Instr.mv(target_reg, reg).with_comment(f"Load variable '{name}'"))
            case LabelStorage(label=label):
                env.emit(
                    Instr.lui(target_reg, f"%hi({label})").with_comment(f"Load variable '{name}' address"),
                    Instr.addi(target_reg, target_reg, f"%lo({label})")
                )


def compile_call(env: CompilerEnv, fun: str, args: typing.List[ast.Expr]):
    """Compile function calls and operators"""

    # Handle arithmetic operators
    if fun in ['+', '-', '*', '/', '%']:
        compile_arithmetic_op(env, fun, args)

    # Handle logical operators
    elif fun in ['&&', '||']:
        compile_logical_op(env, fun, args)

    # Handle comparison operators
    elif fun in ['==', '!=', '<', '<=', '>', '>=']:
        if len(args) != 2:
            raise ValueError(f"Comparison operator {fun} requires exactly 2 arguments")
        compile_comparison(env, fun, args)

    # Handle unary operators
    elif fun == '!':
        compile_unary_op(env, fun, args)

    # Handle built-in functions
    elif fun in ['print_int', 'print_bool', 'print_string']:
        compile_builtin_function(env, fun, args)

    # Handle user-defined function calls
    else:
        compile_function_call(env, fun, args)


def compile_comparison(env: CompilerEnv, op: str, args: typing.List[ast.Expr]):
    """Compile comparison operators"""
    left_target = env.target
    compile_expr(env, args[0])
    left_reg = env.get_target_reg()

    env.target += 1
    compile_expr(env, args[1])
    right_reg = env.get_target_reg()

    env.target = left_target
    target_reg = env.get_target_reg()

    # Use simple register operations instead of branches
    match op:
        case '==':
            env.emit(Instr.sub(target_reg, left_reg, right_reg).with_comment("Compute difference"))
            env.emit(Instr.seqz(target_reg, target_reg).with_comment("Set if equal (difference is zero)"))
        case '!=':
            env.emit(Instr.sub(target_reg, left_reg, right_reg).with_comment("Compute difference"))
            env.emit(Instr.snez(target_reg, target_reg).with_comment("Set if not equal (difference is non-zero)"))
        case '<':
            env.emit(Instr.slt(target_reg, left_reg, right_reg).with_comment("Set if less than"))
        case '<=':
            # a <= b  iff  !(a > b)  iff  !(b < a)
            env.emit(Instr.slt(target_reg, right_reg, left_reg).with_comment("Set if b < a"))
            env.emit(Instr.seqz(target_reg, target_reg).with_comment("Invert to get a <= b"))
        case '>':
            env.emit(Instr.slt(target_reg, right_reg, left_reg).with_comment("Set if greater than"))
        case '>=':
            # a >= b  iff  !(a < b)
            env.emit(Instr.slt(target_reg, left_reg, right_reg).with_comment("Set if a < b"))
            env.emit(Instr.seqz(target_reg, target_reg).with_comment("Invert to get a >= b"))


def compile_stmt(env: CompilerEnv, stmt: ast.Stmt):
    """Compile a statement"""
    match stmt:
        case ast.SExpr(expr=expr):
            # Statement expression - compile but ignore result
            compile_expr(env, expr)

        case ast.VarDecl(typ=typ, name=name, value=value):
            if not is_void_type(typ):
                if value:
                    # Compile initialization value
                    compile_expr(env, value)
                    value_reg = env.get_target_reg()

                    # Allocate storage for variable
                    var_reg = env.allocate_var_register()
                    env.add_variable(name, RegStorage(var_reg))

                    # Move value to variable storage
                    env.emit(Instr.mv(var_reg, value_reg).with_comment(f"Store variable '{name}'"))
                else:
                    # Uninitialized variable - just allocate storage
                    var_reg = env.allocate_var_register()
                    env.add_variable(name, RegStorage(var_reg))

        case ast.Assg(name=name, value=value):
            storage = env.lookup_variable(name)
            if storage is None:
                raise ValueError(f"Variable '{name}' not found for assignment")

            # Compile value
            compile_expr(env, value)
            value_reg = env.get_target_reg()

            # Store to variable
            match storage:
                case RegStorage(reg=reg):
                    env.emit(Instr.mv(reg, value_reg).with_comment(f"Assign to variable '{name}'"))
                case _:
                    raise ValueError(f"Unsupported storage type for assignment to '{name}'")

        case ast.Return(expr=expr):
            if expr:
                # Compile return expression
                compile_expr(env, expr)
                result_reg = env.get_target_reg()
                # Move result to return register
                env.emit(Instr.mv(Reg.A0, result_reg).with_comment("Move return value"))
            else:
                # Void return
                env.emit(Instr.li(Reg.A0, 0).with_comment("Void return"))

            # Simple return
            env.emit(Instr.ret().with_comment("Return from function"))

        case ast.If(cond=cond, then_block=then_block, else_block=else_block):
            compile_if(env, cond, then_block, else_block)

        case _:
            raise ValueError(f"Unsupported statement type: {type(stmt)}")


def compile_if(env: CompilerEnv, cond: ast.Expr, then_block: ast.Block, else_block: typing.Optional[ast.Block] = None):
    """Compile if statement"""
    # Save current block before compiling condition
    original_block = env.current_block

    # Compile condition
    compile_expr(env, cond)
    cond_reg = env.get_target_reg()

    # Restore current block after condition compilation
    env.current_block = original_block

    if else_block:
        # if-then-else
        else_label = env.new_label("if_else")
        end_label = env.new_label("if_end")

        # Branch to else if condition is false (zero)
        env.emit(Instr.beqz(cond_reg, else_label).with_comment("Branch to else if condition is false"))

        # Then block
        compile_block(env, then_block)

        # Only emit jump if the then block didn't end with a breaking instruction
        if not env.current_block.is_closed():
            env.emit(Instr.j(end_label).with_comment("Jump to end after then block"))

        # Create else block
        else_riscv_block = riscv.Block(name=else_label)
        env.current_function.blocks[else_label] = else_riscv_block
        env.current_block = else_riscv_block

        compile_block(env, else_block)

        # Create end block
        end_riscv_block = riscv.Block(name=end_label)
        env.current_function.blocks[end_label] = end_riscv_block
        env.current_block = end_riscv_block
    else:
        # if-then only
        end_label = env.new_label("if_end")

        # Branch to end if condition is false
        env.emit(Instr.beqz(cond_reg, end_label).with_comment("Branch to end if condition is false"))

        # Then block
        compile_block(env, then_block)

        # Create end block
        end_riscv_block = riscv.Block(name=end_label)
        env.current_function.blocks[end_label] = end_riscv_block
        env.current_block = end_riscv_block


def compile_block(env: CompilerEnv, block: ast.Block):
    """Compile a block of statements with proper scoping"""
    # Push a new scope for this block
    env.push_scope()

    try:
        for stmt in block.stmts:
            compile_stmt(env, stmt)
    finally:
        # Always pop the scope, even if compilation fails
        env.pop_scope()


def save_registers(env, regs):
    """Save registers onto the stack in order"""

    # Update the stack pointer to make space
    env.emit(
        Instr.addi(Reg.SP, Reg.SP, -4 * len(regs))
        .with_comment(f"Saving registers: {regs}")
    )

    # Save the registers
    for i, reg in enumerate(regs):
        env.emit(Instr.sw(reg, i * 4, Reg.SP))


def restore_registers(env, regs):
    """Restore registers from stack. Assumes they were saved in order with
    increasing offset from the stack pointer."""

    # Load the registers
    for i, reg in enumerate(regs):
        instr = Instr.lw(reg, i * 4, Reg.SP)
        if i == 0:
            instr = instr.with_comment(f"Restoring registers: {regs}")
        env.emit(instr)

    # Update the stack pointer to free space
    env.emit(Instr.addi(Reg.SP, Reg.SP, 4 * len(regs)))


def compile_function(env: CompilerEnv, fdecl: ast.FunDecl):
    """Compile a function declaration"""
    is_main = fdecl.name == "main"
    env.start_function(fdecl.name, fdecl.args, is_main)

    # # Initialize stack pointer for main function
    # if is_main:
    #     env.emit(Instr.li(Reg.SP, 0x10010000).with_comment("Initialize stack pointer"))

    # Compile function body
    compile_block(env, fdecl.body)

    # Ensure functions have proper return
    if not env.current_block.is_closed():
        if is_main:
            # Main function: use exit system call
            if is_void_type(fdecl.ret):
                env.emit(Instr.li(Reg.A0, 0).with_comment("Exit code 0"))
            env.emit(
                Instr.li(Reg.A7, riscv.SysCall.EXIT).with_comment("Exit system call"),
                Instr.ecall().with_comment("Exit program")
            )
        elif is_void_type(fdecl.ret):
            # Function epilogue for void functions
            env.emit(Instr.ret().with_comment("Return from void function"))
        else:
            # Non-void function without explicit return - return default value
            env.emit(
                Instr.li(Reg.A0, 0).with_comment("Default return value"),
                Instr.ret().with_comment("Return default value")
            )

    env.end_function()


def compile_program(program: ast.Program) -> riscv.Program:
    """Compile a complete program"""
    env = CompilerEnv(riscv.Program())

    # Compile all function declarations, but put main function last
    main_func = None
    other_funcs = []

    for decl in program.decls:
        match decl:
            case ast.FunDecl() if decl.name == "main":
                main_func = decl
            case ast.FunDecl():
                other_funcs.append(decl)

    # Compile main function first
    if main_func:
        compile_function(env, main_func)

    # Compile non-main functions after
    for decl in other_funcs:
        compile_function(env, decl)

    return env.program
