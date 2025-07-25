#!/usr/bin/env python3
"""
Comprehensive test runner for Kompy language with RISC-V backend.
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

from wuwiak import parser as p
from wuwiak import typechecker as t
from wuwiak import compiler_riscv as c


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
        self.current_dir = test_dir / "current"  # For current syntax tests
        self.rars_jar = project_root / "rars" / "rars.jar"

    def setup(self):
        """Setup RARS jar for execution"""
        if not self.rars_jar.exists():
            print(f"Warning: RARS jar not found at {self.rars_jar}")
            return False
        return True

    def run_good_test(self, test_file: pathlib.Path) -> TestResult:
        """
        Run a positive test case - should compile and run successfully.
        For tests with expected output, check that the output matches.
        """
        try:
            # Create a valid module name
            module_name = test_file.stem
            if module_name[0].isdigit():
                module_name = "Test" + module_name
            module_name = module_name.replace("-", "_")

            # Parse the test file
            content = test_file.read_text(encoding='utf-8')
            program = p.program(module_name=module_name).parse(content)

            # Type check
            program_t = t.tc_program(t.init_env(), program)

            # Compile to RISC-V
            riscv_program = c.compile_program(program_t)

            # Generate RISC-V assembly
            assembly_lines = []
            assembly_lines.append(".text")
            assembly_lines.append(".globl main")
            assembly_lines.append("")
            
            # Add data section if present
            if riscv_program.data_section:
                assembly_lines.append(".data")
                for data_directive in riscv_program.data_section:
                    assembly_lines.append(data_directive)
                assembly_lines.append("")
                assembly_lines.append(".text")
            
            # Add functions
            for func in riscv_program.text_section:
                assembly_lines.append(f"{func.name}:")
                
                # Process all blocks in the function
                main_block = func.blocks.get('main')
                if main_block:
                    for instr in main_block.instructions:
                        if instr.operands:
                            operand_str = ', '.join(str(op) for op in instr.operands)
                            assembly_lines.append(f"    {instr.opcode} {operand_str}")
                        else:
                            assembly_lines.append(f"    {instr.opcode}")
                
                # Handle other blocks (for control flow)
                for block_name, block in func.blocks.items():
                    if block_name != 'main':
                        assembly_lines.append(f"{block_name}:")
                        for instr in block.instructions:
                            if instr.operands:
                                operand_str = ', '.join(str(op) for op in instr.operands)
                                assembly_lines.append(f"    {instr.opcode} {operand_str}")
                            else:
                                assembly_lines.append(f"    {instr.opcode}")
                
                assembly_lines.append("")

            # Write assembly file
            asm_file = test_file.with_suffix('.s')
            with open(asm_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(assembly_lines))

            # Run with RARS
            if self.rars_jar.exists():
                try:
                    # Run RARS in console mode 
                    result = subprocess.run([
                        "java", "-jar", str(self.rars_jar),
                        str(asm_file)
                    ], capture_output=True, text=True, timeout=10)

                    # Check if test has expected output
                    expected_output = self._get_expected_output(test_file)
                    if expected_output is not None:
                        # RARS output includes some header text, extract just the program output
                        lines = result.stdout.split('\n')
                        program_output_lines = []
                        capture_output = False
                        
                        for line in lines:
                            # Skip RARS header
                            if "RARS" in line and "Copyright" in line:
                                continue
                            # Look for actual program output (numbers, strings, etc.)
                            if line.strip() and not line.startswith("Program terminated"):
                                program_output_lines.append(line.strip())
                        
                        actual_output = '\n'.join(program_output_lines) if program_output_lines else ""
                        
                        if actual_output == expected_output:
                            return TestResult(test_file.name, True, "Output matches expected", actual_output)
                        else:
                            return TestResult(test_file.name, False,
                                            f"Expected: {expected_output!r}, Got: {actual_output!r}")
                    else:
                        # No expected output, check if program ran without error
                        if "Program terminated by calling exit" in result.stdout or result.returncode == 0:
                            return TestResult(test_file.name, True, "Compiled and ran successfully", result.stdout)
                        else:
                            return TestResult(test_file.name, False,
                                            f"Runtime error: {result.stderr}")
                            
                except subprocess.TimeoutExpired:
                    return TestResult(test_file.name, False, "Program timed out")
                except subprocess.CalledProcessError as e:
                    return TestResult(test_file.name, False, f"RARS execution failed: {e}")
            else:
                # Can't run RARS, just check that it compiles
                return TestResult(test_file.name, True, "Compiled to RISC-V successfully")

        except parsy.ParseError as e:
            return TestResult(test_file.name, False, f"Parse error: {e}")
        except t.TypecheckError as e:
            return TestResult(test_file.name, False, f"Type error: {e}")
        except Exception as e:
            return TestResult(test_file.name, False, f"Fatal error: {e}\n{''.join(traceback.format_exception(e))}")
        finally:
            # Clean up generated files
            for ext in ['.s']:
                cleanup_file = test_file.with_suffix(ext)
                if cleanup_file.exists():
                    cleanup_file.unlink()

    def run_bad_test(self, test_file: pathlib.Path) -> TestResult:
        """
        Run a negative test case - should fail to parse, type check, or compile.
        """
        try:
            # Create a valid module name
            module_name = test_file.stem
            if module_name[0].isdigit():
                module_name = "Test" + module_name
            module_name = module_name.replace("-", "_")

            # Try to parse
            content = test_file.read_text(encoding='utf-8')
            program = p.program(module_name=module_name).parse(content)

            # Try to type check
            program_t = t.tc_program(t.init_env(), program)

            # Try to compile to RISC-V
            riscv_program = c.compile_program(program_t)

            # If we get here, the test should have failed but didn't
            return TestResult(test_file.name, False, "Test should have failed but passed")

        except t.TypecheckError as e:
            # This is expected for bad tests
            return TestResult(test_file.name, True, f"Failed (T) as expected: {e}")
        except parsy.ParseError as e:
            # This is expected for bad tests
            return TestResult(test_file.name, True, f"Failed (P) as expected: {e}")
        except Exception as e:
            # Other compilation errors are also expected
            return TestResult(test_file.name, True, f"Failed (C) as expected: {e}")

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

        # Run current tests (should pass)
        if self.current_dir.exists():
            for test_file in sorted(self.current_dir.glob("*.src")):
                print(f"Running current test: {test_file.name}")
                result = self.run_good_test(test_file)
                good_results.append(result)

        # Run bad tests
        if self.bad_dir.exists():
            for test_file in sorted(self.bad_dir.glob("*.src")):
                print(f"Running bad test: {test_file.name}")
                result = self.run_bad_test(test_file)
                bad_results.append(result)

        return good_results, bad_results

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
