"""
Test runner for the simple compiler
"""
import sys
import pathlib
import argparse

from . import parser as p
from . import typechecker as t
from . import comp


def build_simple(src_file, output=None, classname=None):
    """Build using the simple compiler"""
    src_file = pathlib.Path(src_file)
    try:
        program = p.parse_file(src_file, name=classname)
    except Exception as e:
        print(f"Parse error: {e}")
        sys.exit(1)

    try:
        program_t = t.tc_program(t.init_env(), program)
    except Exception as e:
        print(f"Type check error: {e}")
        sys.exit(1)

    try:
        jvm_class = comp.compile_program(program_t)
    except Exception as e:
        print(f"Compilation error: {e}")
        sys.exit(1)

    jvm_str = jvm_class.gen()
    j_file = output if output else src_file.with_suffix('.j')
    with open(j_file, "w", encoding='utf-8') as f:
        f.write(jvm_str)
    
    print(f"Compiled {src_file} to {j_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file", type=pathlib.Path)
    parser.add_argument("-o", type=pathlib.Path, required=False)
    parser.add_argument("-c", type=str, required=False)
    
    args = parser.parse_args()
    build_simple(args.src_file, output=args.o, classname=args.c)
