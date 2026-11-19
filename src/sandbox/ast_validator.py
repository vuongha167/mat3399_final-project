# -*- coding: utf-8 -*-
"""
Trình phân tích và kiểm tra an toàn cây cú pháp trừu tượng AST (Abstract Syntax Tree).
Ngăn chặn các lệnh nguy hiểm (truy cập hệ thống, I/O tệp, socket mạng, thực thi động)
trước khi mã được chuyển vào môi trường thực thi SymPy.
"""

import ast
import re
from typing import Tuple, List, Set


# Danh sách các module bị cấm hoàn toàn
DISALLOWED_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "http", "ftplib", "builtins", "__builtin__", "importlib", "pkgutil",
    "pathlib", "fcntl", "pty", "posix", "nt", "signal", "ctypes", "multiprocessing",
    "threading", "asyncio", "resource", "gc", "inspect", "pdb"
}

# Danh sách các hàm dựng sẵn (built-in functions) bị cấm
DISALLOWED_CALLS: Set[str] = {
    "eval", "exec", "open", "compile", "__import__",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "breakpoint", "exit", "quit",
    "input", "memoryview"
}

# Danh sách các thuộc tính nội tại (dunder attributes) tiềm ẩn lỗ hổng leo thang
DISALLOWED_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__bases__", "__mro__", "__globals__",
    "__code__", "__closure__", "__class__", "__builtins__",
    "__dict__", "__module__", "__annotations__"
}

# Danh sách các module an toàn được phép import
ALLOWED_MODULES: Set[str] = {
    "sympy", "math", "cmath", "fractions", "decimal"
}


def extract_code_block(text: str) -> str:
    """
    Trích xuất mã Python từ chuỗi phản hồi của LLM (thường nằm trong markdown code block).
    Nếu không có khối mã markdown, trả về chuỗi đã được loại bỏ khoảng trắng dư thừa.
    """
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # Lấy khối mã dài nhất hoặc khối cuối cùng
        return matches[-1].strip()
    return text.strip()


class SecurityASTVisitor(ast.NodeVisitor):
    """
    Duyệt cây AST để phát hiện bất kỳ cấu trúc cú pháp nào vi phạm chính sách bảo mật.
    """

    def __init__(self, max_nodes: int = 1000):
        self.max_nodes = max_nodes
        self.node_count = 0
        self.violations: List[str] = []

    def generic_visit(self, node: ast.AST):
        self.node_count += 1
        if self.node_count > self.max_nodes:
            self.violations.append(
                f"Độ phức tạp vượt ngưỡng an toàn: số nút AST ({self.node_count}) > {self.max_nodes}"
            )
            return
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in DISALLOWED_MODULES or base_module not in ALLOWED_MODULES:
                self.violations.append(
                    f"Cấm import module không an toàn: '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in DISALLOWED_MODULES or base_module not in ALLOWED_MODULES:
                self.violations.append(
                    f"Cấm import từ module không an toàn: '{node.module}'"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Kiểm tra tên hàm gọi trực tiếp
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DISALLOWED_CALLS:
                self.violations.append(f"Cấm gọi hàm nguy hiểm: '{func_name}()'")
        
        # Kiểm tra gọi phương thức hoặc thuộc tính
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in DISALLOWED_CALLS or attr_name in DISALLOWED_ATTRIBUTES:
                self.violations.append(f"Cấm gọi hàm/thuộc tính nguy hiểm: '{attr_name}'")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in DISALLOWED_ATTRIBUTES:
            self.violations.append(f"Cấm truy cập thuộc tính nội tại: '{node.attr}'")
        self.generic_visit(node)


def validate_python_code(
    code: str,
    max_nodes: int = 1000,
    max_length: int = 4000
) -> Tuple[bool, str, str]:
    """
    Kiểm tra tính an toàn của đoạn mã Python trước khi thực thi.

    Args:
        code: Mã Python nguồn hoặc văn bản chứa khối mã.
        max_nodes: Giới hạn tối đa số nút AST.
        max_length: Giới hạn tối đa số ký tự trong mã.

    Returns:
        Tuple (is_safe, reason, sanitized_code):
            - is_safe: True nếu mã an toàn và đúng cú pháp.
            - reason: Lý do vi phạm hoặc lỗi cú pháp nếu không an toàn, rỗng nếu an toàn.
            - sanitized_code: Đoạn mã đã được trích xuất sạch sẽ.
    """
    sanitized_code = extract_code_block(code)

    if not sanitized_code:
        return False, "Mã nguồn rỗng hoặc không tìm thấy khối mã Python hợp lệ.", ""

    if len(sanitized_code) > max_length:
        return (
            False,
            f"Kích thước mã vượt quá giới hạn ({len(sanitized_code)} > {max_length} ký tự).",
            sanitized_code
        )

    try:
        tree = ast.parse(sanitized_code)
    except SyntaxError as err:
        return False, f"Lỗi cú pháp Python (SyntaxError): {err.msg} tại dòng {err.lineno}", sanitized_code
    except Exception as exc:
        return False, f"Không thể phân tích cây cú pháp AST: {str(exc)}", sanitized_code

    visitor = SecurityASTVisitor(max_nodes=max_nodes)
    visitor.visit(tree)

    if visitor.violations:
        return False, "; ".join(visitor.violations), sanitized_code

    return True, "", sanitized_code
