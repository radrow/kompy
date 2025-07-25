#!/usr/bin/env python3
"""
Kompy Compiler Main Entry Point

This module serves as the main entry point for the kompy compiler.
It provides a command-line interface with two main commands:
- build: compile source files to RISC-V assembly
- run: compile and execute using the RARS simulator

Usage:
    python -m kompy build <source_file>
    python -m kompy run <source_file>
"""

import argparse
import sys
import subprocess
import typing
import traceback
from pathlib import Path

from . import ast, parser, error, typechecker as t, compiler_riscv


def build_command(source_file: str, output_file: typing.Optional[str] = None) -> str:
    """
    Build command: compile source file to RISC-V assembly

    Args:
        source_file: Path to the source file (.src)
        output_file: Optional output file path (.s)

    Returns:
        Path to the generated assembly file
    """
    source_path = Path(source_file)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    # Determine output file
    if output_file is None:
        output_path = source_path.with_suffix('.s')
    else:
        output_path = Path(output_file)

    try:
        print(f"Parsing {source_file}...")
        # Parse the source file
        prog = parser.parse_file(source_path)

        print("Type checking...")
        # Type check
        env = t.init_env()
        typed_prog = t.tc_program(env, prog)

        print("Compiling to RISC-V assembly...")
        # Compile to RISC-V
        riscv_prog_obj = compiler_riscv.compile_program(typed_prog)
        riscv_assembly = riscv_prog_obj.gen()

        # Write assembly to file
        print(f"Writing assembly to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(riscv_assembly)

        print(f"✓ Compilation successful! Assembly written to {output_path}")
        return str(output_path)

    except error.WuwiakError as e:
        print(f"✗ Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(-1)


def run_command(source_file: str) -> None:
    """
    Run command: compile source file and execute with RARS simulator

    Args:
        source_file: Path to the source file (.src)
    """
    print(f"Building {source_file}...")

    # First, build the assembly file
    assembly_file = build_command(source_file)

    # Find the rars.jar file
    kompy_root = Path(__file__).parent.parent.parent
    rars_jar = kompy_root / "rars" / "rars.jar"

    if not rars_jar.exists():
        print(f"✗ RARS simulator not found at {rars_jar}", file=sys.stderr)
        print("Please make sure rars.jar is available in the rars/ directory", file=sys.stderr)
        sys.exit(1)

    print(f"Running assembly with RARS simulator...")
    print(f"Assembly file: {assembly_file}")
    print("-" * 50)

    try:
        # Run RARS with the assembly file
        # Use headless mode for command line execution
        result = subprocess.run([
            "java", "-jar", str(rars_jar),
            str(assembly_file)
        ], text=True, timeout=30)  # Add timeout to prevent hanging

        print("-" * 50)
        if result.returncode == 0:
            print("✓ Program executed successfully")
        else:
            print(f"✗ Program exited with code {result.returncode}")
            sys.exit(result.returncode)

    except subprocess.TimeoutExpired:
        print("✗ Program execution timed out (30 seconds)", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"✗ RARS execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Java not found. Please make sure Java is installed and in your PATH", file=sys.stderr)
        print("RARS requires Java to run", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the kompy compiler"""
    try:
        parser = argparse.ArgumentParser(
            prog='kompy',
            description='Kompy Programming Language Compiler',
            epilog="""
Examples:
  python -m kompy build program.src           # Compile to program.s
  python -m kompy build program.src -o out.s  # Compile to out.s
  python -m kompy run program.src             # Compile and run with RARS
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            required=True
        )

        # Build command
        build_parser = subparsers.add_parser(
            'build',
            help='Compile source file to RISC-V assembly'
        )
        build_parser.add_argument(
            'source_file',
            help='Source file to compile (.src)'
        )
        build_parser.add_argument(
            '-o', '--output',
            help='Output assembly file (.s). Defaults to source file with .s extension'
        )

        # Run command
        run_parser = subparsers.add_parser(
            'run',
            help='Compile and run source file with RARS simulator'
        )
        run_parser.add_argument(
            'source_file',
            help='Source file to compile and run (.src)'
        )

        # Parse arguments
        args = parser.parse_args()

        # Dispatch to appropriate command
        if args.command == 'build':
            build_command(args.source_file, args.output)
        elif args.command == 'run':
            run_command(args.source_file)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user", file=sys.stderr)
        sys.exit(1)
    except SystemExit:
        # Let argparse handle its own exits
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
