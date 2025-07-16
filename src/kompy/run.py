"""
Helper module to work with the jasmin assembler
"""
import os
import sys
import pathlib
import subprocess
import re
import argparse

import parsy

from . import parser as p
from . import typechecker as t
from . import compiler as c


def find_project_root():
    try:
        return pathlib.Path(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            stderr=subprocess.DEVNULL
        ).decode().strip())
    except IOError:
        return pathlib.Path(__file__).resolve().parent


def ensure_jasmin():
    root = find_project_root()
    jasmin_path = root / 'jasmin' / 'jasmin.jar'

    if not jasmin_path.is_file():
        url = "https://github.com/davidar/jasmin.git"
        target_dir = "jasmin"

        if not os.path.exists(target_dir):
            subprocess.run(["git", "clone", url, target_dir], check=True)

    return jasmin_path


def assemble(j_file, jasmin_jar=None):
    if not jasmin_jar:
        jasmin_jar = ensure_jasmin()

    result = subprocess.run(
        ["java", "-jar", jasmin_jar, j_file],
        check=True,
        capture_output=True,
        text=True,
    )

    match = re.search(r"Generated: (.*\.class)", result.stdout)
    return match.group(1) if match else None


def eval_class(classfile):
    # Remove .class extension to get the class name
    classname = str(classfile).replace('.class', '')
    result = subprocess.run(
        ["java", classname],
        check=True,
        capture_output=True,
        text=True,
    )

    return result


def eval_j(j_file, jasmin_jar=None):
    classfile = assemble(j_file, jasmin_jar=jasmin_jar)
    return eval_class(classfile)


def build(src_file, output=None):
    src_file = pathlib.Path(src_file)
    try:
        program = p.parse_file(src_file)
    except parsy.ParseError as e:
        print(e)
        sys.exit(1)

    program_t = t.tc_program(t.init_env(), program)
    jvm_class = c.compile_program(program_t)

    jvm_str = jvm_class.gen()
    j_file = output if output else src_file.with_suffix('.j')
    with open(j_file, "w", encoding='utf-8') as f:
        f.write(jvm_str)


def main():
    a = argparse.ArgumentParser()
    subparsers = a.add_subparsers(dest="command", required=True)

    a_build = subparsers.add_parser("build")
    a_build.add_argument("src_file", type=pathlib.Path)
    a_build.add_argument("-o", type=pathlib.Path, required=False)
    a_build.set_defaults(func=lambda args: build(args.src_file, output=args.o))

    args = a.parse_args()
    args.func(args)
