#!/usr/bin/env python3
"""
Comprehensive test runner for Kompy language.
Tests are divided into 'good' (should compile and run) and 'bad' (should fail).
"""

import os
import sys
import pathlib
import subprocess
import tempfile
import shutil
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple

import parsy

from kompy import parser as p
from kompy import typechecker as t
from kompy import compiler as c
from kompy import run


# Add src to path to import kompy modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Check if we're in a virtual environment, if not, try to activate it
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    venv_path = project_root / '.venv'
    if venv_path.exists():
        # Add venv packages to path
        venv_packages = venv_path / 'lib' / 'python3.13' / 'site-packages'
        if venv_packages.exists():
            sys.path.insert(0, str(venv_packages))

FAIL_STR = "\x1b[31;1mFAIL\x1b[0m"
PASS_STR = "\x1b[32;1mPASS\x1b[0m"


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", output: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.output = output


class TestRunner:
    def __init__(self, test_dir: pathlib.Path):
        self.test_dir = test_dir
        self.good_dir = test_dir / "good"
        self.bad_dir = test_dir / "bad"
        self.jasmin_jar = None

    def setup(self):
        """Setup jasmin jar for compilation"""
        try:
            self.jasmin_jar = run.ensure_jasmin()
        except Exception as e:
            print(f"Warning: Could not setup jasmin: {e}")

    def run_good_test(self, test_file: pathlib.Path) -> TestResult:
        """
        Run a positive test case - should compile and run successfully.
        For tests with expected output, check that the output matches.
        """
        try:
            # Create a valid module name for Java (can't start with digit)
            module_name = test_file.stem
            if module_name[0].isdigit():
                module_name = "Test" + module_name
            module_name = module_name.replace("-", "_")

            # Parse the test file with the proper module name
            content = test_file.read_text(encoding='utf-8')
            program = p.program(module_name=module_name).parse(content)

            # Type check
            program_t = t.tc_program(t.init_env(), program)

            # Compile to JVM
            jvm_class = c.compile_program(program_t)

            # Find the main function (if it exists)
            main_function = None
            for decl in program.decls:
                if isinstance(decl, p.ast.FunDecl) and decl.name == "main":
                    main_function = decl
                    break

            if main_function is None:
                return TestResult(test_file.name, False, "No main function found")

            # Check if main function has the correct signature
            # If it's already void main([string] args), we don't need to add a Java wrapper
            needs_wrapper = True
            if (main_function.ret.name == "void" and
                len(main_function.args) == 1 and
                isinstance(main_function.args[0][0], p.ast.TypeArr) and
                main_function.args[0][0].el.name == "string" and
                main_function.args[0][1] == "args"):
                needs_wrapper = False

            # Add a proper Java main method that calls our main function (only if needed)
            if needs_wrapper:
                main_method = self._create_java_main_method(main_function, module_name)
                jvm_class.methods.append(main_method)

            jvm_str = jvm_class.gen()

            # Write jasmin file
            j_file = test_file.with_suffix('.j')
            with open(j_file, 'w', encoding='utf-8') as f:
                f.write(jvm_str)

            # Assemble with jasmin
            if self.jasmin_jar:
                try:
                    # Need to run jasmin in the same directory as the .j file
                    # so the .class file is generated in the right place
                    old_cwd = os.getcwd()
                    os.chdir(test_file.parent)
                    try:
                        classfile = run.assemble(str(j_file.name))
                    finally:
                        os.chdir(old_cwd)

                    # Run the program
                    result = subprocess.run(
                        ["java", pathlib.Path(classfile).stem],
                        cwd=test_file.parent,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    # Check if test has expected output
                    expected_output = self._get_expected_output(test_file)
                    if expected_output is not None:
                        # For string comparisons, we need to handle escape sequences
                        actual_output = result.stdout.rstrip('\n')  # Only remove trailing newline
                        if actual_output == expected_output:
                            return TestResult(test_file.name, True, "Output matches expected", result.stdout)
                        else:
                            return TestResult(test_file.name, False,
                                            f"Expected: {expected_output!r}, Got: {actual_output!r}")
                    else:
                        # No expected output, just check that it ran without error
                        if result.returncode == 0:
                            return TestResult(test_file.name, True, "Compiled and ran successfully", result.stdout)
                        else:
                            return TestResult(test_file.name, False,
                                            f"Runtime error: {result.stderr}")
                except subprocess.CalledProcessError as e:
                    return TestResult(test_file.name, False, f"Jasmin assembly failed: {e}")
            else:
                # Can't run jasmin, just check that it compiles
                return TestResult(test_file.name, True, "Compiled to JVM successfully")

        except parsy.ParseError as e:
            return TestResult(test_file.name, False, f"Parse error: {e}")
        except t.TypecheckError as e:
            return TestResult(test_file.name, False, f"Type error: {e}")
        except Exception as e:
            return TestResult(test_file.name, False, f"Fatal error: {e}\n{''.join(traceback.format_exception(e))}")
        finally:
            # Clean up generated files
            for ext in ['.j', '.class']:
                cleanup_file = test_file.with_suffix(ext)
                if cleanup_file.exists():
                    cleanup_file.unlink()

    def _create_java_main_method(self, main_function, class_name):
        """Create a proper Java main method that calls our main function and prints the result"""
        from kompy.jvm import Method, Block, Instr, INIT_BLOCK

        # Get the return type
        ret_type = main_function.ret.name

        # Create instructions to call our main function and print the result to stdout
        if ret_type == "int":
            # For int return type, call the function and print the result
            instructions = [
                Instr.getstatic("java/lang/System/out", "Ljava/io/PrintStream;"),
                Instr.invokestatic(f"{class_name}/main()I"),
                Instr.invokevirtual("java/io/PrintStream/println(I)V"),
                Instr.return_()
            ]
        elif ret_type == "bool":
            # For bool return type, call the function and print the result (0 or 1)
            instructions = [
                Instr.getstatic("java/lang/System/out", "Ljava/io/PrintStream;"),
                Instr.invokestatic(f"{class_name}/main()I"),
                Instr.invokevirtual("java/io/PrintStream/println(I)V"),
                Instr.return_()
            ]
        elif ret_type == "string":
            # For string return type, call the function and print the string
            instructions = [
                Instr.getstatic("java/lang/System/out", "Ljava/io/PrintStream;"),
                Instr.invokestatic(f"{class_name}/main()Ljava/lang/String;"),
                Instr.invokevirtual("java/io/PrintStream/println(Ljava/lang/String;)V"),
                Instr.return_()
            ]
        else:
            # For void or other types, just call the function
            instructions = [
                Instr.invokestatic(f"{class_name}/main()V"),
                Instr.return_()
            ]

        # Create the method
        method = Method(
            visibility='public',
            name='main',
            static=True,
            args=['[Ljava/lang/String;'],
            ret='V',
            stack=1000,
            local=1000,
            blocks={INIT_BLOCK: Block(instructions=instructions)}
        )

        return method

    def run_bad_test(self, test_file: pathlib.Path) -> TestResult:
        """
        Run a negative test case - should fail to parse, type check, or compile.
        """
        try:
            # Create a valid module name for Java (can't start with digit)
            module_name = test_file.stem
            if module_name[0].isdigit():
                module_name = "Test" + module_name
            module_name = module_name.replace("-", "_")

            # Try to parse
            content = test_file.read_text(encoding='utf-8')
            program = p.program(module_name=module_name).parse(content)

            # Try to type check
            program_t = t.tc_program(t.init_env(), program)

            # Try to compile
            jvm_class = c.compile_program(program_t)
            jvm_str = jvm_class.gen()

            # If we get here, the test should have failed but didn't
            return TestResult(test_file.name, False, "Test should have failed but passed")

        except t.TypecheckError as e:
            # This is expected for bad tests
            return TestResult(test_file.name, True, f"Failed (T) as expected: {e}")
        except parsy.ParseError as e:
            # This is expected for bad tests
            return TestResult(test_file.name, True, f"Failed (P) as expected: {e}")

    def _get_expected_output(self, test_file: pathlib.Path) -> Optional[str]:
        """
        Extract expected output from test file comments.
        Looks for comments like: # Expected: Hello World
        """
        try:
            content = test_file.read_text(encoding='utf-8')
            match = re.search(r'#\s*Expected:\s*(.+)', content)
            if match:
                expected = match.group(1)
                # Handle escape sequences in expected output
                expected = expected.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\\\', '\\')
                return expected
        except:
            pass
        return None

    def run_all_tests(self) -> Tuple[List[TestResult], List[TestResult]]:
        """Run all tests and return results"""
        good_results = []
        bad_results = []

        # Run good tests
        if self.good_dir.exists():
            for test_file in sorted(self.good_dir.glob("*.src")):
                print(f"Running good test: {test_file.name}")
                result = self.run_good_test(test_file)
                good_results.append(result)

        # Run bad tests
        if self.bad_dir.exists():
            for test_file in sorted(self.bad_dir.glob("*.src")):
                print(f"Running bad test: {test_file.name}")
                result = self.run_bad_test(test_file)
                bad_results.append(result)

        return good_results, bad_results

    def print_results(self, good_results: List[TestResult], bad_results: List[TestResult]):
        """Print test results summary"""
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)

        good_passed = sum(1 for r in good_results if r.passed)
        bad_passed = sum(1 for r in bad_results if r.passed)

        print(f"\nGood Tests: {good_passed}/{len(good_results)} passed")
        for result in good_results:
            status = PASS_STR if result.passed else FAIL_STR
            print(f"  {status}: {result.name} - {result.message}")
            if result.output:
                print(f"    Output: {result.output}")

        print(f"\nBad Tests: {bad_passed}/{len(bad_results)} passed")
        for result in bad_results:
            status = PASS_STR if result.passed else FAIL_STR
            print(f"  {status}: {result.name} - {result.message}")

        total_passed = good_passed + bad_passed
        total_tests = len(good_results) + len(bad_results)
        print(f"\nOverall: {total_passed}/{total_tests} tests passed")

        return total_passed == total_tests


def main():
    test_dir = pathlib.Path(__file__).parent
    runner = TestRunner(test_dir)
    runner.setup()

    good_results, bad_results = runner.run_all_tests()
    success = runner.print_results(good_results, bad_results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
