"""
Kompy Compiler Runtime Module

This module provides the main runtime functionality for the Kompy compiler,
including compilation, assembly with Jasmin, and execution of Kompy programs.
"""

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Optional, Union

import parsy

from . import compiler as c
from . import parser as p
from . import typechecker as t


class KompyError(Exception):
    """Base exception for Kompy compiler errors."""
    pass


class JasminError(KompyError):
    """Exception raised when Jasmin assembly fails."""
    pass


class ExecutionError(KompyError):
    """Exception raised when Java class execution fails."""
    pass


class CompilationError(KompyError):
    """Exception raised when Kompy compilation fails."""
    pass


class ProjectManager:
    """Manages project-level operations like finding the root and dependencies."""

    JASMIN_REPO_URL = "https://github.com/davidar/jasmin.git"
    JASMIN_JAR_PATH = "jasmin/jasmin.jar"

    def __init__(self):
        self._project_root = None
        self._jasmin_path = None

    @property
    def project_root(self) -> pathlib.Path:
        """Get the project root directory."""
        if self._project_root is None:
            self._project_root = self._find_project_root()
        return self._project_root

    def _find_project_root(self) -> pathlib.Path:
        """Find the project root directory."""
        try:
            git_root = subprocess.check_output(
                ['git', 'rev-parse', '--show-toplevel'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return pathlib.Path(git_root)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return pathlib.Path(__file__).resolve().parent

    @property
    def jasmin_jar(self) -> pathlib.Path:
        """Get the path to Jasmin JAR, downloading if necessary."""
        if self._jasmin_path is None:
            self._jasmin_path = self._ensure_jasmin()
        return self._jasmin_path

    def _ensure_jasmin(self) -> pathlib.Path:
        """Ensure Jasmin assembler is available."""
        jasmin_path = self.project_root / self.JASMIN_JAR_PATH

        if jasmin_path.is_file():
            return jasmin_path

        jasmin_dir = self.project_root / "jasmin"
        if not jasmin_dir.exists():
            subprocess.run(
                ["git", "clone", self.JASMIN_REPO_URL, str(jasmin_dir)],
                check=True,
                capture_output=True,
                text=True
            )

        if not jasmin_path.is_file():
            raise FileNotFoundError(f"Jasmin JAR not found at: {jasmin_path}")

        return jasmin_path


class JasminAssembler:
    """Handles Jasmin assembly operations."""

    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager

    def assemble(self, j_file: Union[str, pathlib.Path]) -> pathlib.Path:
        """Assemble a Jasmin file into a Java class file."""
        j_file_path = pathlib.Path(j_file)

        if not j_file_path.exists():
            raise FileNotFoundError(f"Jasmin file not found: {j_file_path}")

        try:
            result = subprocess.run(
                ["java", "-jar", str(self.project_manager.jasmin_jar), str(j_file_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            class_file_match = re.search(r"Generated: (.*\.class)", result.stdout)
            if not class_file_match:
                raise JasminError("Assembly completed but could not determine output file")

            return pathlib.Path(class_file_match.group(1))

        except subprocess.CalledProcessError as e:
            raise JasminError(f"Assembly failed: {e.stderr}")


class JavaExecutor:
    """Handles Java class execution."""

    def execute_class(self, class_file: Union[str, pathlib.Path]) -> subprocess.CompletedProcess:
        """Execute a compiled Java class file."""
        class_path = pathlib.Path(class_file)

        if not class_path.exists():
            raise FileNotFoundError(f"Class file not found: {class_path}")

        class_name = class_path.stem

        try:
            return subprocess.run(
                ["java", class_name],
                check=True,
                capture_output=True,
                text=True,
                cwd=class_path.parent
            )
        except subprocess.CalledProcessError as e:
            raise ExecutionError(f"Execution failed: {e.stderr}")

    def execute_jasmin_file(self, j_file: Union[str, pathlib.Path],
                           assembler: JasminAssembler) -> subprocess.CompletedProcess:
        """Assemble and execute a Jasmin file in one step."""
        class_file = assembler.assemble(j_file)
        return self.execute_class(class_file)


class KompyCompiler:
    """Main compiler class that orchestrates the compilation process."""

    def __init__(self):
        self.project_manager = ProjectManager()
        self.assembler = JasminAssembler(self.project_manager)
        self.executor = JavaExecutor()

    def compile_to_jasmin(self, src_file: Union[str, pathlib.Path],
                         output: Optional[Union[str, pathlib.Path]] = None,
                         class_name: Optional[str] = None) -> pathlib.Path:
        """Compile a Kompy source file to Jasmin assembly."""
        src_path = pathlib.Path(src_file)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        try:
            # Parse, type-check, and compile
            program = p.parse_file(src_path, name=class_name)
            program_typed = t.tc_program(t.init_env(), program)
            jvm_class = c.compile_program(program_typed)
            jvm_assembly = jvm_class.gen()

            # Write output
            output_path = pathlib.Path(output) if output else src_path.with_suffix('.j')
            output_path.write_text(jvm_assembly, encoding='utf-8')

            return output_path

        except parsy.ParseError as e:
            raise CompilationError(f"Parse error: {e}")
        except Exception as e:
            raise CompilationError(f"Compilation failed: {e}")

    def compile_and_assemble(self, src_file: Union[str, pathlib.Path],
                           class_name: Optional[str] = None) -> pathlib.Path:
        """Compile Kompy source to Java class file."""
        jasmin_file = self.compile_to_jasmin(src_file, class_name=class_name)
        return self.assembler.assemble(jasmin_file)

    def compile_and_run(self, src_file: Union[str, pathlib.Path],
                       class_name: Optional[str] = None) -> subprocess.CompletedProcess:
        """Compile and execute a Kompy source file."""
        class_file = self.compile_and_assemble(src_file, class_name=class_name)
        return self.executor.execute_class(class_file)


class CLIHandler:
    """Handles command-line interface operations."""

    def __init__(self):
        self.compiler = KompyCompiler()

    def handle_build(self, src_file: pathlib.Path, output: Optional[pathlib.Path] = None,
                    class_name: Optional[str] = None) -> None:
        """Handle the build command."""
        try:
            output_path = self.compiler.compile_to_jasmin(src_file, output, class_name)
            print(f"Successfully compiled {src_file} -> {output_path}")
        except (CompilationError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def handle_run(self, src_file: pathlib.Path, class_name: Optional[str] = None) -> None:
        """Handle the run command."""
        try:
            result = self.compiler.compile_and_run(src_file, class_name)
            print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='')
        except (CompilationError, JasminError, ExecutionError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def handle_assemble(self, j_file: pathlib.Path) -> None:
        """Handle the assemble command."""
        try:
            class_file = self.compiler.assembler.assemble(j_file)
            print(f"Successfully assembled {j_file} -> {class_file}")
        except (JasminError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='kompy',
        description='Kompy Programming Language Compiler and Runtime'
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help='Available commands')

    # Build command
    build_parser = subparsers.add_parser("build", help="Compile Kompy source to Jasmin assembly")
    build_parser.add_argument("src_file", type=pathlib.Path, help="Kompy source file")
    build_parser.add_argument("-o", "--output", type=pathlib.Path, help="Output Jasmin file")
    build_parser.add_argument("-c", "--class-name", type=str, help="Java class name")

    # Run command
    run_parser = subparsers.add_parser("run", help="Compile and execute Kompy source")
    run_parser.add_argument("src_file", type=pathlib.Path, help="Kompy source file")
    run_parser.add_argument("-c", "--class-name", type=str, help="Java class name")

    # Assemble command
    assemble_parser = subparsers.add_parser("assemble", help="Assemble Jasmin file to Java class")
    assemble_parser.add_argument("j_file", type=pathlib.Path, help="Jasmin assembly file")

    return parser


def main() -> None:
    """Main entry point for the Kompy compiler CLI."""
    parser = create_argument_parser()
    cli = CLIHandler()

    try:
        args = parser.parse_args()

        if args.command == "build":
            cli.handle_build(args.src_file, args.output, args.class_name)
        elif args.command == "run":
            cli.handle_run(args.src_file, args.class_name)
        elif args.command == "assemble":
            cli.handle_assemble(args.j_file)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


# Backward compatibility functions (deprecated)
def build(
        src_file: Union[str, pathlib.Path],
        output: Optional[Union[str, pathlib.Path]] = None,
        class_name: Optional[str] = None
) -> None:
    """Deprecated: Use KompyCompiler.compile_to_jasmin instead."""
    compiler = KompyCompiler()
    try:
        compiler.compile_to_jasmin(src_file, output, class_name)
    except (CompilationError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def assemble(j_file: Union[str, pathlib.Path]) -> Optional[str]:
    """Deprecated: Use JasminAssembler.assemble instead."""
    try:
        project_manager = ProjectManager()
        assembler = JasminAssembler(project_manager)
        class_file = assembler.assemble(j_file)
        return str(class_file)
    except (JasminError, FileNotFoundError):
        return None


if __name__ == "__main__":
    main()
