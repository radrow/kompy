"""
Representation of the Java Virtual Machine assembly
"""
import typing
import dataclasses
from dataclasses import dataclass
import json

from attrs import mutable, field


INIT_BLOCK = '$init'
INDENT = '    '


@dataclass
class Instr:
    """
    Single JVM instruction
    """
    opcode: str
    operands: typing.List[typing.Union[int, str]] = dataclasses.field(default_factory=list)
    branching: typing.Optional[str] = None
    breaking: bool = False
    stack_delta: typing.Union[int, typing.List[int]] = 0

    def __str__(self) -> str:
        s = self.opcode
        for op in self.operands:
            # For ldc with string operands, we need to quote the string
            if self.opcode == "ldc" and isinstance(op, str):
                s += f" {json.dumps(op)}"
            else:
                s += f" {op}"
        return s

    # --- Literal / Constant instructions ---

    @classmethod
    def iconst(cls, val: int) -> "Instr":
        if -1 <= val <= 5:
            # Some `iconst` have fixed variants
            return cls(f"iconst_{val}", stack_delta=1)
        if -128 <= val <= 127:
            # Bytes
            return cls("bipush", [val], stack_delta=1)
        if -32768 <= val <= 32767:
            # Short int
            return cls("sipush", [val], stack_delta=1)
        return cls("ldc", [val], stack_delta=1)

    @classmethod
    def aconst_null(cls) -> "Instr":
        return cls("aconst_null", stack_delta=1)

    @classmethod
    def ldc(cls, value: typing.Union[str, int]) -> "Instr":
        return cls("ldc", [value], stack_delta=1)

    # --- Stack operations ---

    @classmethod
    def pop(cls) -> "Instr":
        return cls("pop", stack_delta=-1)

    @classmethod
    def dup(cls) -> "Instr":
        return cls("dup", stack_delta=1)

    @classmethod
    def dup2(cls) -> "Instr":
        return cls("dup2", stack_delta=[1, 2])

    @classmethod
    def swap(cls) -> "Instr":
        return cls("swap")

    # --- Local variable load/store ---

    @classmethod
    def iload(cls, index: int) -> "Instr":
        return cls(f"iload_{index}" if index <= 3 else "iload",
                   [index] if index > 3 else [],
                   stack_delta=1,
                   )

    @classmethod
    def istore(cls, index: int) -> "Instr":
        return cls(f"istore_{index}" if index <= 3 else "istore",
                   [index] if index > 3 else [],
                   stack_delta=-1,
                   )

    @classmethod
    def aload(cls, index: int) -> "Instr":
        return cls(f"aload_{index}" if index <= 3 else "aload",
                   [index] if index > 3 else [],
                   stack_delta=1,
                   )

    @classmethod
    def astore(cls, index: int) -> "Instr":
        return cls(f"astore_{index}" if index <= 3 else "astore",
                   [index] if index > 3 else [],
                   stack_delta=-1,
                   )

    # --- Arithmetic (int) ---

    @classmethod
    def iadd(cls) -> "Instr":
        return cls("iadd", stack_delta=-1)

    @classmethod
    def isub(cls) -> "Instr":
        return cls("isub", stack_delta=-1)

    @classmethod
    def imul(cls) -> "Instr":
        return cls("imul", stack_delta=-1)

    @classmethod
    def idiv(cls) -> "Instr":
        return cls("idiv", stack_delta=-1)

    @classmethod
    def irem(cls) -> "Instr":
        return cls("irem", stack_delta=-1)

    @classmethod
    def ineg(cls) -> "Instr":
        return cls("ineg", stack_delta=0)

    @classmethod
    def iand(cls) -> "Instr":
        return cls("iand", stack_delta=-1)

    @classmethod
    def ior(cls) -> "Instr":
        return cls("ior", stack_delta=-1)

    @classmethod
    def ixor(cls) -> "Instr":
        return cls("ixor", stack_delta=-1)

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
        return cls("ifeq", [label], branching=label, stack_delta=-1)

    @classmethod
    def ifne(cls, label: str) -> "Instr":
        return cls("ifne", [label], branching=label, stack_delta=-1)

    @classmethod
    def iflt(cls, label: str) -> "Instr":
        return cls("iflt", [label], branching=label, stack_delta=-1)

    @classmethod
    def ifle(cls, label: str) -> "Instr":
        return cls("ifle", [label], branching=label, stack_delta=-1)

    @classmethod
    def ifgt(cls, label: str) -> "Instr":
        return cls("ifgt", [label], branching=label, stack_delta=-1)

    @classmethod
    def ifge(cls, label: str) -> "Instr":
        return cls("ifge", [label], branching=label, stack_delta=-1)

    # --- Conditional jumps (2-value int comparisons) ---

    @classmethod
    def if_icmpeq(cls, label: str) -> "Instr":
        return cls("if_icmpeq", [label], branching=label, stack_delta=-2)

    @classmethod
    def if_icmpne(cls, label: str) -> "Instr":
        return cls("if_icmpne", [label], branching=label, stack_delta=-2)

    @classmethod
    def if_icmplt(cls, label: str) -> "Instr":
        return cls("if_icmplt", [label], branching=label, stack_delta=-2)

    @classmethod
    def if_icmpgt(cls, label: str) -> "Instr":
        return cls("if_icmpgt", [label], branching=label, stack_delta=-2)

    @classmethod
    def if_icmple(cls, label: str) -> "Instr":
        return cls("if_icmple", [label], branching=label, stack_delta=-2)

    @classmethod
    def if_icmpge(cls, label: str) -> "Instr":
        return cls("if_icmpge", [label], branching=label, stack_delta=-2)

    @classmethod
    def goto(cls, label: str) -> "Instr":
        return cls("goto", [label], branching=label, breaking=True)

    # --- Return ---

    @classmethod
    def ireturn(cls) -> "Instr":
        return cls("ireturn", breaking=True, stack_delta=-1)

    @classmethod
    def areturn(cls) -> "Instr":
        return cls("areturn", breaking=True, stack_delta=-1)

    @classmethod
    def return_(cls) -> "Instr":
        return cls("return", breaking=True, stack_delta=-1)

    # --- Method call ---

    @classmethod
    def invokevirtual(cls, method: str) -> "Instr":
        #  TODO get type and compute based on it
        return cls("invokevirtual", [method], stack_delta=[])

    @classmethod
    def invokestatic(cls, method: str) -> "Instr":
        return cls("invokestatic", [method], stack_delta=[])

    @classmethod
    def invokespecial(cls, method: str) -> "Instr":
        return cls("invokespecial", [method], stack_delta=[])

    # --- Object creation ---

    @classmethod
    def new(cls, typename: str) -> "Instr":
        return cls("new", [typename], stack_delta=1)

    @classmethod
    def getstatic(cls, src: str, typename: str) -> "Instr":
        return cls("getstatic", [src, typename], stack_delta=1)

    # --- Raw instruction ---

    # @classmethod
    # def raw(cls,
    #         opcode: str,
    #         operands: typing.Optional[typing.List[typing.Union[int, str]]] = None,
    #         breaking: bool = False,
    #         branching: typing.Optional[str] = None,
    #         stack_delta: typing.Union[int, typing.List[int]] = 0,
    #         ) -> "Instr":
    #     operands = operands if operands else []
    #     return cls(opcode,
    #                operands,
    #                breaking=breaking,
    #                branching=branching,
    #                stack_delta=stack_delta
    #                )


@mutable(auto_attribs=True, kw_only=True)
class Block:
    """
    Block is a sequence of instructions
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
class Method:
    visibility: typing.Literal['public', 'private', 'protected', '']
    name: str
    static: bool
    args: typing.List[str] = field(factory=list)
    ret: str
    stack: int
    local: int
    blocks: typing.Dict[str, Block] = field(factory=dict)

    def gen(self):
        lines = []
        lines.append(f'.method {self.visibility}{" static" if self.static else ""} {self.name}({"".join(self.args)}){self.ret}')
        lines.append(f'.limit stack {self.stack}')
        lines.append(f'.limit locals {self.local}')

        blocks = self.blocks.copy()
        init = blocks.pop(INIT_BLOCK)

        lines.append('')
        lines.append(f'{INIT_BLOCK}:')
        lines.append(init.gen())

        for label, block in blocks.items():
            lines.append('')
            lines.append(f'{label}:')
            lines.append(block.gen())

        lines.append('.end method')
        return '\n'.join(lines) + '\n'


@mutable(auto_attribs=True, kw_only=True)
class Class:
    """
    JVM class
    """
    name: str
    visibility: typing.Literal['public', 'private', 'protected', '']
    superclass: str
    methods: typing.List[Method] = field(factory=list)

    def gen(self):
        lines = []
        lines.append(f'.class {self.visibility} {self.name}')
        lines.append(f'.super {self.superclass}')
        for method in self.methods:
            lines.append('')
            lines.append(method.gen())
        return '\n'.join(lines)
