# -*- coding: utf-8 -*-
"""
Kiểm thử đơn vị (Unit Tests) cho Trình phân tích cây cú pháp trừu tượng AST (ast_validator.py).
Đảm bảo phát hiện và loại bỏ mọi mẫu mã độc hại, truy xuất trái phép hệ thống.
"""

import unittest
from src.sandbox.ast_validator import validate_python_code, extract_code_block


class TestASTValidator(unittest.TestCase):

    def test_extract_code_block_markdown(self):
        text = "Dưới đây là mã Python:\n```python\nx = 10\nprint(x)\n```\nChúc bạn thành công!"
        extracted = extract_code_block(text)
        self.assertEqual(extracted, "x = 10\nprint(x)")

    def test_safe_pure_math_code(self):
        code = """
import math
a = 15
b = 20
c = math.sqrt(a**2 + b**2)
result = c
"""
        is_safe, reason, sanitized = validate_python_code(code)
        self.assertTrue(is_safe, f"Mã an toàn nhưng bị chặn: {reason}")
        self.assertEqual(reason, "")

    def test_safe_sympy_code(self):
        code = """
from sympy import symbols, Eq, solve
x, y = symbols('x y')
eq1 = Eq(2*x + y, 10)
eq2 = Eq(x - y, 2)
result = solve((eq1, eq2), (x, y))
"""
        is_safe, reason, sanitized = validate_python_code(code)
        self.assertTrue(is_safe, f"Mã SymPy an toàn bị chặn: {reason}")

    def test_block_os_import(self):
        code = "import os\nos.system('rm -rf /')"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm import module không an toàn: 'os'", reason)

    def test_block_subprocess_import(self):
        code = "from subprocess import Popen\nPopen(['ls'])"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm import từ module không an toàn: 'subprocess'", reason)

    def test_block_eval_call(self):
        code = "user_input = '1+1'\nval = eval(user_input)"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm gọi hàm nguy hiểm: 'eval()'", reason)

    def test_block_exec_call(self):
        code = "exec('print(123)')"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm gọi hàm nguy hiểm: 'exec()'", reason)

    def test_block_open_call(self):
        code = "f = open('/etc/passwd', 'r')"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm gọi hàm nguy hiểm: 'open()'", reason)

    def test_block_subclasses_traversal(self):
        code = "cls = ().__class__.__bases__[0].__subclasses__()"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Cấm truy cập thuộc tính nội tại", reason)

    def test_syntax_error_handling(self):
        code = "def invalid_syntax(: print('error')"
        is_safe, reason, _ = validate_python_code(code)
        self.assertFalse(is_safe)
        self.assertIn("Lỗi cú pháp Python", reason)


if __name__ == "__main__":
    unittest.main()
