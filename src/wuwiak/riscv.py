"""
Representation of RISC-V assembly (RV32IMF: integers, multiplication, floats)
"""
import typing
import dataclasses
from dataclasses import dataclass

from attrs import mutable, field


INIT_BLOCK = 'main'
INDENT = '    '


@dataclass
class Instr:
    """
    Single RISC-V instruction
    """
    opcode: str
    operands: typing.List[typing.Union[int, str]] = dataclasses.field(default_factory=list)
    branching: typing.Optional[str] = None
    breaking: bool = False
    comment: typing.Optional[str] = None

    def __str__(self) -> str:
        s = self.opcode
        if self.operands:
            s += " " + ", ".join(str(op) for op in self.operands)
        if self.comment:
            s += f"  # {self.comment}"
        return s

    def with_comment(self, comment: str) -> "Instr":
        """Return a copy of this instruction with a comment attached"""
        return dataclasses.replace(self, comment=comment)

    @classmethod
    def nil(cls) -> "Instr":
        return cls("", [])

    # --- Load immediate and constants ---

    @classmethod
    def li(cls, rd: str, imm: int) -> "Instr":
        """Load immediate"""
        return cls("li", [rd, imm])

    @classmethod
    def la(cls, rd: str, label: str) -> "Instr":
        """Load address"""
        return cls("la", [rd, label])

    @classmethod
    def lui(cls, rd: str, imm: int) -> "Instr":
        """Load upper immediate"""
        return cls("lui", [rd, imm])

    @classmethod
    def auipc(cls, rd: str, imm: int) -> "Instr":
        """Add upper immediate to PC"""
        return cls("auipc", [rd, imm])

    # --- Arithmetic operations (R-type) ---

    @classmethod
    def add(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Add"""
        return cls("add", [rd, rs1, rs2])

    @classmethod
    def sub(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Subtract"""
        return cls("sub", [rd, rs1, rs2])

    @classmethod
    def mul(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Multiply (M extension)"""
        return cls("mul", [rd, rs1, rs2])

    @classmethod
    def div(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Divide (M extension)"""
        return cls("div", [rd, rs1, rs2])

    @classmethod
    def rem(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Remainder (M extension)"""
        return cls("rem", [rd, rs1, rs2])

    # --- Arithmetic operations (I-type) ---

    @classmethod
    def addi(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """Add immediate"""
        return cls("addi", [rd, rs1, imm])

    @classmethod
    def slti(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """Set less than immediate"""
        return cls("slti", [rd, rs1, imm])

    @classmethod
    def sltiu(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """Set less than immediate unsigned"""
        return cls("sltiu", [rd, rs1, imm])

    @classmethod
    def xori(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """XOR immediate"""
        return cls("xori", [rd, rs1, imm])

    @classmethod
    def ori(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """OR immediate"""
        return cls("ori", [rd, rs1, imm])

    @classmethod
    def andi(cls, rd: str, rs1: str, imm: int) -> "Instr":
        """AND immediate"""
        return cls("andi", [rd, rs1, imm])

    # --- Logical operations ---

    @classmethod
    def sll(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Shift left logical"""
        return cls("sll", [rd, rs1, rs2])

    @classmethod
    def srl(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Shift right logical"""
        return cls("srl", [rd, rs1, rs2])

    @classmethod
    def sra(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Shift right arithmetic"""
        return cls("sra", [rd, rs1, rs2])

    @classmethod
    def slli(cls, rd: str, rs1: str, shamt: int) -> "Instr":
        """Shift left logical immediate"""
        return cls("slli", [rd, rs1, shamt])

    @classmethod
    def srli(cls, rd: str, rs1: str, shamt: int) -> "Instr":
        """Shift right logical immediate"""
        return cls("srli", [rd, rs1, shamt])

    @classmethod
    def srai(cls, rd: str, rs1: str, shamt: int) -> "Instr":
        """Shift right arithmetic immediate"""
        return cls("srai", [rd, rs1, shamt])

    @classmethod
    def and_(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Bitwise AND"""
        return cls("and", [rd, rs1, rs2])

    @classmethod
    def or_(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Bitwise OR"""
        return cls("or", [rd, rs1, rs2])

    @classmethod
    def xor(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Bitwise XOR"""
        return cls("xor", [rd, rs1, rs2])

    @classmethod
    def slt(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Set less than"""
        return cls("slt", [rd, rs1, rs2])

    @classmethod
    def sltu(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Set less than unsigned"""
        return cls("sltu", [rd, rs1, rs2])

    # --- Load/Store operations ---

    @classmethod
    def lw(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load word"""
        return cls("lw", [rd, f"{offset}({rs1})"])

    @classmethod
    def lh(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load halfword"""
        return cls("lh", [rd, f"{offset}({rs1})"])

    @classmethod
    def lhu(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load halfword unsigned"""
        return cls("lhu", [rd, f"{offset}({rs1})"])

    @classmethod
    def lb(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load byte"""
        return cls("lb", [rd, f"{offset}({rs1})"])

    @classmethod
    def lbu(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load byte unsigned"""
        return cls("lbu", [rd, f"{offset}({rs1})"])

    @classmethod
    def sw(cls, rs2: str, offset: int, rs1: str) -> "Instr":
        """Store word"""
        return cls("sw", [rs2, f"{offset}({rs1})"])

    @classmethod
    def sh(cls, rs2: str, offset: int, rs1: str) -> "Instr":
        """Store halfword"""
        return cls("sh", [rs2, f"{offset}({rs1})"])

    @classmethod
    def sb(cls, rs2: str, offset: int, rs1: str) -> "Instr":
        """Store byte"""
        return cls("sb", [rs2, f"{offset}({rs1})"])

    # --- Branch operations ---

    @classmethod
    def beq(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if equal"""
        return cls("beq", [rs1, rs2, label], branching=label)

    @classmethod
    def bne(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if not equal"""
        return cls("bne", [rs1, rs2, label], branching=label)

    @classmethod
    def blt(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if less than"""
        return cls("blt", [rs1, rs2, label], branching=label)

    @classmethod
    def bge(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if greater or equal"""
        return cls("bge", [rs1, rs2, label], branching=label)

    @classmethod
    def bltu(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if less than unsigned"""
        return cls("bltu", [rs1, rs2, label], branching=label)

    @classmethod
    def bgeu(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if greater or equal unsigned"""
        return cls("bgeu", [rs1, rs2, label], branching=label)

    # --- Pseudo-instructions for branches ---

    @classmethod
    def bgt(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if greater than (pseudo-instruction for blt rs2, rs1, label)"""
        return cls("bgt", [rs1, rs2, label], branching=label)

    @classmethod
    def ble(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if less than or equal (pseudo-instruction for bge rs2, rs1, label)"""
        return cls("ble", [rs1, rs2, label], branching=label)

    @classmethod
    def bgtu(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if greater than unsigned (pseudo-instruction for bltu rs2, rs1, label)"""
        return cls("bgtu", [rs1, rs2, label], branching=label)

    @classmethod
    def bleu(cls, rs1: str, rs2: str, label: str) -> "Instr":
        """Branch if less than or equal unsigned (pseudo-instruction for bgeu rs2, rs1, label)"""
        return cls("bleu", [rs1, rs2, label], branching=label)

    @classmethod
    def beqz(cls, rs1: str, label: str) -> "Instr":
        """Branch if equal to zero (pseudo-instruction for beq rs1, x0, label)"""
        return cls("beqz", [rs1, label], branching=label)

    @classmethod
    def bnez(cls, rs1: str, label: str) -> "Instr":
        """Branch if not equal to zero (pseudo-instruction for bne rs1, x0, label)"""
        return cls("bnez", [rs1, label], branching=label)

    # --- Jump operations ---

    @classmethod
    def jal(cls, rd: str, label: str) -> "Instr":
        """Jump and link"""
        return cls("jal", [rd, label], branching=label)

    @classmethod
    def jalr(cls, rd: str, rs1: str, imm: int = 0) -> "Instr":
        """Jump and link register"""
        return cls("jalr", [rd, rs1, imm])

    @classmethod
    def j(cls, label: str) -> "Instr":
        """Jump (pseudo-instruction for jal x0, label)"""
        return cls("j", [label], branching=label, breaking=True)

    @classmethod
    def jr(cls, rs1: str) -> "Instr":
        """Jump register (pseudo-instruction for jalr x0, rs1, 0)"""
        return cls("jr", [rs1], breaking=True)

    @classmethod
    def ret(cls) -> "Instr":
        """Return (pseudo-instruction for jalr x0, ra, 0)"""
        return cls("ret", [], breaking=True)

    # --- Floating-point operations (F extension) ---

    @classmethod
    def flw(cls, rd: str, offset: int, rs1: str) -> "Instr":
        """Load single-precision float"""
        return cls("flw", [rd, f"{offset}({rs1})"])

    @classmethod
    def fsw(cls, rs2: str, offset: int, rs1: str) -> "Instr":
        """Store single-precision float"""
        return cls("fsw", [rs2, f"{offset}({rs1})"])

    @classmethod
    def fadd_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Add single-precision floats"""
        return cls("fadd.s", [rd, rs1, rs2])

    @classmethod
    def fsub_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Subtract single-precision floats"""
        return cls("fsub.s", [rd, rs1, rs2])

    @classmethod
    def fmul_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Multiply single-precision floats"""
        return cls("fmul.s", [rd, rs1, rs2])

    @classmethod
    def fdiv_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Divide single-precision floats"""
        return cls("fdiv.s", [rd, rs1, rs2])

    @classmethod
    def fsqrt_s(cls, rd: str, rs1: str) -> "Instr":
        """Square root single-precision float"""
        return cls("fsqrt.s", [rd, rs1])

    @classmethod
    def fmin_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Minimum single-precision float"""
        return cls("fmin.s", [rd, rs1, rs2])

    @classmethod
    def fmax_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Maximum single-precision float"""
        return cls("fmax.s", [rd, rs1, rs2])

    # --- Float comparisons ---

    @classmethod
    def feq_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Float equal"""
        return cls("feq.s", [rd, rs1, rs2])

    @classmethod
    def flt_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Float less than"""
        return cls("flt.s", [rd, rs1, rs2])

    @classmethod
    def fle_s(cls, rd: str, rs1: str, rs2: str) -> "Instr":
        """Float less than or equal"""
        return cls("fle.s", [rd, rs1, rs2])

    # --- Float conversions ---

    @classmethod
    def fcvt_w_s(cls, rd: str, rs1: str) -> "Instr":
        """Convert float to int"""
        return cls("fcvt.w.s", [rd, rs1])

    @classmethod
    def fcvt_s_w(cls, rd: str, rs1: str) -> "Instr":
        """Convert int to float"""
        return cls("fcvt.s.w", [rd, rs1])

    @classmethod
    def fmv_w_x(cls, rd: str, rs1: str) -> "Instr":
        """Move from integer register to float register"""
        return cls("fmv.w.x", [rd, rs1])

    @classmethod
    def fmv_x_w(cls, rd: str, rs1: str) -> "Instr":
        """Move from float register to integer register"""
        return cls("fmv.x.w", [rd, rs1])

    # --- Pseudo-instructions ---

    @classmethod
    def nop(cls) -> "Instr":
        """No operation (pseudo-instruction for addi x0, x0, 0)"""
        return cls("nop", [])

    @classmethod
    def mv(cls, rd: str, rs1: str) -> "Instr":
        """Move (pseudo-instruction for addi rd, rs1, 0)"""
        return cls("mv", [rd, rs1])

    @classmethod
    def not_(cls, rd: str, rs1: str) -> "Instr":
        """Bitwise NOT (pseudo-instruction for xori rd, rs1, -1)"""
        return cls("not", [rd, rs1])

    @classmethod
    def neg(cls, rd: str, rs1: str) -> "Instr":
        """Negate (pseudo-instruction for sub rd, x0, rs1)"""
        return cls("neg", [rd, rs1])

    @classmethod
    def seqz(cls, rd: str, rs1: str) -> "Instr":
        """Set if equal to zero (pseudo-instruction for sltiu rd, rs1, 1)"""
        return cls("seqz", [rd, rs1])

    @classmethod
    def snez(cls, rd: str, rs1: str) -> "Instr":
        """Set if not equal to zero (pseudo-instruction for sltu rd, x0, rs1)"""
        return cls("snez", [rd, rs1])

    @classmethod
    def sgtz(cls, rd: str, rs1: str) -> "Instr":
        """Set if greater than zero (pseudo-instruction for slt rd, x0, rs1)"""
        return cls("sgtz", [rd, rs1])

    @classmethod
    def sltz(cls, rd: str, rs1: str) -> "Instr":
        """Set if less than zero (pseudo-instruction for slt rd, rs1, x0)"""
        return cls("sltz", [rd, rs1])

    @classmethod
    def label(cls, name: str) -> "Instr":
        """Create a label (not an instruction, but a marker)"""
        return cls("", [f"{name}:"])

    # --- System calls ---

    @classmethod
    def ecall(cls) -> "Instr":
        """Environment call (system call)"""
        return cls("ecall", [])

    @classmethod
    def exit_ecall(cls) -> "Instr":
        """Environment call for exit (breaks execution)"""
        return cls("ecall", [], breaking=True)

    @classmethod
    def ebreak(cls) -> "Instr":
        """Environment break (breakpoint)"""
        return cls("ebreak", [])


@mutable(auto_attribs=True, kw_only=True)
class Block:
    """
    Block is a sequence of instructions with an optional label
    """
    name: str
    instructions: typing.List[Instr] = field(factory=list)
    closed: bool = False

    def append(self, *instrs: Instr):
        if self.is_closed():
            raise ValueError(f"Appending to a closed block {self.name}")

        for i in instrs[0:-1]:
            if i.breaking:
                raise ValueError(f"Illegal breaking instruction in the middle {self.name}")

        self.instructions.extend(instrs)

    def is_closed(self):
        return self.instructions != [] and self.instructions[-1].breaking

    def gen(self):
        lines = []
        for instr in self.instructions:
            lines.append(f'{INDENT}{instr}')
        return '\n'.join(lines)


@mutable(auto_attribs=True, kw_only=True)
class Function:
    """
    RISC-V function/procedure
    """
    name: str
    global_: bool = True
    blocks: typing.Dict[str, Block] = field(factory=dict)

    def gen(self):
        lines = []

        if self.global_:
            lines.append(f'.globl {self.name}')

        lines.append(f'{self.name}:')

        blocks = self.blocks.copy()
        if INIT_BLOCK in blocks:
            init = blocks.pop(INIT_BLOCK)
            lines.append(init.gen())

        for label, block in blocks.items():
            lines.append(f'{label}:')
            lines.append(block.gen())

        return '\n'.join(lines) + '\n'


@mutable(auto_attribs=True, kw_only=True)
class Program:
    """
    RISC-V program
    """
    data_section: typing.List[str] = field(factory=list)
    text_section: typing.List[Function] = field(factory=list)

    def add_data(self, directive: str):
        """Add a directive to the data section"""
        self.data_section.append(directive)

    def add_function(self, function: Function):
        """Add a function to the text section"""
        self.text_section.append(function)

    def gen(self):
        lines = []

        # Data section
        if self.data_section:
            lines.append('.data')
            for directive in self.data_section:
                lines.append(directive)
            lines.append('')

        # Text section
        lines.append('.text')
        for function in self.text_section:
            lines.append('')
            lines.append(function.gen())

        return '\n'.join(lines)


# Common register names for convenience
class Reg:
    """RISC-V register names"""
    # Integer registers
    ZERO = "x0"    # Hard-wired zero
    RA = "ra"      # Return address (x1)
    SP = "sp"      # Stack pointer (x2)
    GP = "gp"      # Global pointer (x3)
    TP = "tp"      # Thread pointer (x4)
    T0 = "t0"      # Temporary (x5)
    T1 = "t1"      # Temporary (x6)
    T2 = "t2"      # Temporary (x7)
    S0 = "s0"      # Saved register / Frame pointer (x8)
    FP = "fp"      # Frame pointer (alias for s0)
    S1 = "s1"      # Saved register (x9)
    A0 = "a0"      # Function argument / return value (x10)
    A1 = "a1"      # Function argument / return value (x11)
    A2 = "a2"      # Function argument (x12)
    A3 = "a3"      # Function argument (x13)
    A4 = "a4"      # Function argument (x14)
    A5 = "a5"      # Function argument (x15)
    A6 = "a6"      # Function argument (x16)
    A7 = "a7"      # Function argument (x17)
    S2 = "s2"      # Saved register (x18)
    S3 = "s3"      # Saved register (x19)
    S4 = "s4"      # Saved register (x20)
    S5 = "s5"      # Saved register (x21)
    S6 = "s6"      # Saved register (x22)
    S7 = "s7"      # Saved register (x23)
    S8 = "s8"      # Saved register (x24)
    S9 = "s9"      # Saved register (x25)
    S10 = "s10"    # Saved register (x26)
    S11 = "s11"    # Saved register (x27)
    T3 = "t3"      # Temporary (x28)
    T4 = "t4"      # Temporary (x29)
    T5 = "t5"      # Temporary (x30)
    T6 = "t6"      # Temporary (x31)

    T_REGS = 7
    A_REGS = 8
    S_REGS = 12

    @classmethod
    def A(cls, i: int):
        assert 0 <= i < cls.A_REGS
        return f'a{i}'

    @classmethod
    def T(cls, i: int):
        assert 0 <= i < cls.T_REGS
        return f't{i}'

    @classmethod
    def S(cls, i: int):
        assert 0 <= i < cls.S_REGS
        return f's{i}'


    # Floating-point registers
    FT0 = "ft0"    # Temporary (f0)
    FT1 = "ft1"    # Temporary (f1)
    FT2 = "ft2"    # Temporary (f2)
    FT3 = "ft3"    # Temporary (f3)
    FT4 = "ft4"    # Temporary (f4)
    FT5 = "ft5"    # Temporary (f5)
    FT6 = "ft6"    # Temporary (f6)
    FT7 = "ft7"    # Temporary (f7)
    FS0 = "fs0"    # Saved (f8)
    FS1 = "fs1"    # Saved (f9)
    FA0 = "fa0"    # Function argument / return value (f10)
    FA1 = "fa1"    # Function argument / return value (f11)
    FA2 = "fa2"    # Function argument (f12)
    FA3 = "fa3"    # Function argument (f13)
    FA4 = "fa4"    # Function argument (f14)
    FA5 = "fa5"    # Function argument (f15)
    FA6 = "fa6"    # Function argument (f16)
    FA7 = "fa7"    # Function argument (f17)
    FS2 = "fs2"    # Saved (f18)
    FS3 = "fs3"    # Saved (f19)
    FS4 = "fs4"    # Saved (f20)
    FS5 = "fs5"    # Saved (f21)
    FS6 = "fs6"    # Saved (f22)
    FS7 = "fs7"    # Saved (f23)
    FS8 = "fs8"    # Saved (f24)
    FS9 = "fs9"    # Saved (f25)
    FS10 = "fs10"  # Saved (f26)
    FS11 = "fs11"  # Saved (f27)
    FT8 = "ft8"    # Temporary (f28)
    FT9 = "ft9"    # Temporary (f29)
    FT10 = "ft10"  # Temporary (f30)
    FT11 = "ft11"  # Temporary (f31)


CALLER_SAVED = [
    Reg.RA,
    Reg.T0, Reg.T1, Reg.T2, Reg.T3, Reg.T4, Reg.T5, Reg.T6,
    Reg.A0, Reg.A1, Reg.A2, Reg.A3, Reg.A4, Reg.A5, Reg.A6,
]

CALLEE_SAVED = [
    Reg.SP,
    Reg.S0, Reg.S1, Reg.S2, Reg.S3, Reg.S4, Reg.S5, Reg.S6,
    Reg.S7, Reg.S8, Reg.S9, Reg.S10, Reg.S11,
]

# System call numbers for common operations
class SysCall:
    """Common RISC-V system call numbers"""
    PRINT_INT = 1
    PRINT_FLOAT = 2
    PRINT_STRING = 4
    READ_INT = 5
    READ_FLOAT = 6
    READ_STRING = 8
    SBRK = 9           # Memory allocation
    EXIT = 10
    PRINT_CHAR = 11
    READ_CHAR = 12
    OPEN_FILE = 13
    READ_FILE = 14
    WRITE_FILE = 15
    CLOSE_FILE = 16
    EXIT2 = 17         # Exit with value in a1
