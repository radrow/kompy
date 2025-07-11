import attr
import typing
from dataclasses import dataclass


INIT_BLOCK = '$init'
INDENT = '    '


@dataclass
class Instr:
    opcode: str
    operand: typing.Optional[typing.Union[int, str]] = None

    def __str__(self) -> str:
        if self.operand is not None:
            return f"{self.opcode} {self.operand}"
        return self.opcode

    # --- Literal / Constant instructions ---

    @classmethod
    def iconst(cls, val: int) -> "Instr":
        if -1 <= val <= 5:
            # Some `iconst` have fixed variants
            return cls(f"iconst_{val}")
        elif -128 <= val <= 127:
            # Bytes
            return cls("bipush", val)
        elif -32768 <= val <= 32767:
            # Short int
            return cls("sipush", val)
        else:
            return cls("ldc", val)

    @classmethod
    def aconst_null(cls) -> "Instr":
        return cls("aconst_null")

    @classmethod
    def ldc(cls, value: typing.Union[str, int]) -> "Instr":
        return cls("ldc", value)

    # --- Stack operations ---

    @classmethod
    def pop(cls) -> "Instr":
        return cls("pop")

    @classmethod
    def dup(cls) -> "Instr":
        return cls("dup")

    @classmethod
    def dup2(cls) -> "Instr":
        return cls("dup2")

    @classmethod
    def swap(cls) -> "Instr":
        return cls("swap")

    # --- Local variable load/store ---

    @classmethod
    def iload(cls, index: int) -> "Instr":
        return cls(f"iload_{index}" if index <= 3 else "iload",
                   index if index > 3 else None
                   )

    @classmethod
    def istore(cls, index: int) -> "Instr":
        return cls(f"istore_{index}" if index <= 3 else "istore",
                   index if index > 3 else None
                   )

    @classmethod
    def aload(cls, index: int) -> "Instr":
        return cls(f"aload_{index}" if index <= 3 else "aload",
                   index if index > 3 else None
                   )

    @classmethod
    def astore(cls, index: int) -> "Instr":
        return cls(f"astore_{index}" if index <= 3 else "astore",
                   index if index > 3 else None
                   )

    # --- Arithmetic (int) ---

    @classmethod
    def iadd(cls) -> "Instr":
        return cls("iadd")

    @classmethod
    def isub(cls) -> "Instr":
        return cls("isub")

    @classmethod
    def imul(cls) -> "Instr":
        return cls("imul")

    @classmethod
    def idiv(cls) -> "Instr":
        return cls("idiv")

    @classmethod
    def irem(cls) -> "Instr":
        return cls("irem")

    @classmethod
    def ineg(cls) -> "Instr":
        return cls("ineg")

    @classmethod
    def iand(cls) -> "Instr":
        return cls("iand")

    @classmethod
    def ior(cls) -> "Instr":
        return cls("ior")

    @classmethod
    def ixor(cls) -> "Instr":
        return cls("ixor")

    # --- Shift ops ---

    @classmethod
    def ishl(cls) -> "Instr":
        return cls("ishl")

    @classmethod
    def ishr(cls) -> "Instr":
        return cls("ishr")

    @classmethod
    def iushr(cls) -> "Instr":
        return cls("iushr")

    # --- Conditional jumps (single value) ---

    @classmethod
    def ifeq(cls, label: str) -> "Instr":
        return cls("ifeq", label)

    @classmethod
    def ifne(cls, label: str) -> "Instr":
        return cls("ifne", label)

    @classmethod
    def iflt(cls, label: str) -> "Instr":
        return cls("iflt", label)

    @classmethod
    def ifle(cls, label: str) -> "Instr":
        return cls("ifle", label)

    @classmethod
    def ifgt(cls, label: str) -> "Instr":
        return cls("ifgt", label)

    @classmethod
    def ifge(cls, label: str) -> "Instr":
        return cls("ifge", label)

    # --- Conditional jumps (2-value int comparisons) ---

    @classmethod
    def if_icmpeq(cls, label: str) -> "Instr":
        return cls("if_icmpeq", label)

    @classmethod
    def if_icmpne(cls, label: str) -> "Instr":
        return cls("if_icmpne", label)

    @classmethod
    def if_icmplt(cls, label: str) -> "Instr":
        return cls("if_icmplt", label)

    @classmethod
    def if_icmpgt(cls, label: str) -> "Instr":
        return cls("if_icmpgt", label)

    @classmethod
    def if_icmple(cls, label: str) -> "Instr":
        return cls("if_icmple", label)

    @classmethod
    def if_icmpge(cls, label: str) -> "Instr":
        return cls("if_icmpge", label)

    @classmethod
    def goto(cls, label: str) -> "Instr":
        return cls("goto", label)

    # --- Return ---

    @classmethod
    def ireturn(cls) -> "Instr":
        return cls("ireturn")

    @classmethod
    def areturn(cls) -> "Instr":
        return cls("areturn")

    @classmethod
    def return_(cls) -> "Instr":
        return cls("return")

    # --- Method call ---

    @classmethod
    def invokevirtual(cls, method: str) -> "Instr":
        return cls("invokevirtual", method)

    @classmethod
    def invokestatic(cls, method: str) -> "Instr":
        return cls("invokestatic", method)

    @classmethod
    def invokespecial(cls, method: str) -> "Instr":
        return cls("invokespecial", method)

    # --- Object creation ---

    @classmethod
    def new(cls, typename: str) -> "Instr":
        return cls("new", typename)

    # --- Raw instruction ---

    @classmethod
    def raw(cls,
            opcode: str,
            operand: typing.Optional[typing.Union[int, str]] = None
            ) -> "Instr":
        return cls(opcode, operand)


@attr.s(auto_attribs=True, kw_only=True)
class Block:
    """
    Block is a sequence of instructions
    """
    instructions: typing.List[Instr] = attr.Factory(list)

    def append(self, instr: typing.Union[Instr, typing.Iterable[Instr]]):
        if isinstance(instr, Instr):
            self.instructions += [instr]
        else:
            self.instructions += instr

    def gen(self):
        lines = []
        for instr in self.instructions:
            lines.append(f'{INDENT}{instr}')
        return '\n'.join(lines)


@attr.s(auto_attribs=True, kw_only=True)
class Method:
    visibility: typing.Literal['public', 'private', 'protected', '']
    name: str
    static: bool
    args: typing.List[str]
    ret: str
    stack: int
    local: int
    blocks: typing.Dict[str, Block] = {}

    def gen(self):
        lines = []
        lines.append(f'.method {self.visibility}{" static" if self.static else ""} {self.name}({"".join(self.args)}){self.ret}')
        lines.append(f'.limit stack {self.stack}')
        lines.append(f'.limit locals {self.local}')

        blocks = self.blocks.copy()
        init = blocks.pop(INIT_BLOCK)

        lines.append('')
        for instr in init:
            lines.append(f'{INDENT}{instr}')

        for label, block in blocks.items():
            lines.append('')
            lines.append(f'{label}:')
            lines.append(block.gen())

        return '\n'.join(lines)


@attr.s(auto_attribs=True, kw_only=True)
class Class:
    name: str
    visibility: typing.Literal['public', 'private', 'protected', '']
    superclass: str
    methods: typing.List[Method]

    def gen(self):
        lines = []
        lines.append(f'.class {self.visibility} {self.name}')
        lines.append(f'.super {self.superclass}')
        for method in self.methods:
            lines.append('')
            lines.append(method.gen())
        return '\n'.join(lines)
