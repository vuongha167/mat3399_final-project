# -*- coding: utf-8 -*-
"""
Kiểm thử đơn vị (Unit Tests) cho Môi trường thực thi Hộp cát (executor.py).
"""

import unittest
from src.sandbox.executor import SafeSandboxExecutor


class TestSafeSandboxExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = SafeSandboxExecutor(timeout=1.0)

    def test_pure_arithmetic_execution(self):
        code = """
a = 10
b = 25
result = a + b * 2
"""
        res = self.executor.execute(code)
        self.assertTrue(res.success)
        self.assertEqual(res.output, 60)
        self.assertIsNone(res.error)

    def test_math_module_execution(self):
        code = """
import math
radius = 7
area = math.pi * (radius ** 2)
result = round(area, 2)
"""
        res = self.executor.execute(code)
        self.assertTrue(res.success, f"Thực thi math module thất bại: {res.error}")
        self.assertAlmostEqual(res.output, 153.94, places=2)

    def test_runtime_zero_division_error(self):
        code = """
x = 10 / 0
result = x
"""
        res = self.executor.execute(code)
        self.assertFalse(res.success)
        self.assertIn("ZeroDivisionError", str(res.error))

    def test_ast_blocked_code_execution(self):
        code = """
import os
os.listdir('.')
"""
        res = self.executor.execute(code)
        self.assertFalse(res.success)
        self.assertFalse(res.ast_safe)
        self.assertIn("AST Security Violation", str(res.error))

    def test_infinite_loop_timeout(self):
        # Kiểm tra bắt timeout chính xác và kill tiến trình con
        code_loop = """
while True:
    pass
"""
        short_executor = SafeSandboxExecutor(timeout=0.3)
        res = short_executor.execute(code_loop)
        self.assertFalse(res.success)
        self.assertIn("Execution Timeout", str(res.error))
        self.assertGreaterEqual(res.execution_time, 0.25)

    def test_linear_system_2d_execution(self):
        """Kiểm thử giải hệ phương trình bậc nhất 2 ẩn trong Sandbox."""
        code = """
from sympy import symbols, Eq, solve
x, y = symbols('x y')
eq1 = Eq(2*x + y, 10)
eq2 = Eq(x - y, 2)
result = solve([eq1, eq2], [x, y])
"""
        res = self.executor.execute(code)
        self.assertTrue(res.success, f"Giải hệ 2 ẩn thất bại: {res.error}")
        self.assertIsInstance(res.output, dict)
        self.assertEqual(res.output.get("x"), 4)
        self.assertEqual(res.output.get("y"), 2)

    def test_math_functions_sqrt_pi_execution(self):
        """Kiểm thử các hàm toán học mở rộng sqrt, pi, Abs, Rational."""
        code = """
from sympy import symbols, Eq, solve, sqrt, pi, Abs, Rational
x = symbols('x')
r = Rational(1, 2)
c = sqrt(16)
eq = Eq(x + c * r, 10)
sols = solve(eq, x)
result = sols[0]
"""
        res = self.executor.execute(code)
        self.assertTrue(res.success, f"Thực thi hàm toán học thất bại: {res.error}")
        self.assertEqual(res.output, 8)


if __name__ == "__main__":
    unittest.main()

