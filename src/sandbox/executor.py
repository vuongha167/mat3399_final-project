# -*- coding: utf-8 -*-
"""
Môi trường thực thi hộp cát an toàn (Safe Sandbox Executor).
Thực thi mã SymPy/Toán học có kiểm soát thời gian chờ (Timeout 3s),
bắt ngoại lệ, cô lập không gian tên và hủy tiến trình con độc lập khi vượt ngưỡng thời gian.
"""

import io
import multiprocessing
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.sandbox.ast_validator import validate_python_code, ALLOWED_MODULES


@dataclass
class ExecutionResult:
    """Đối tượng chuẩn hóa kết quả thực thi trong Sandbox."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    execution_time: float = 0.0
    sanitized_code: str = ""
    ast_safe: bool = True
    ast_violation: Optional[str] = None


def _format_value(val: Any) -> Any:
    """
    Chuyển đổi kiểu dữ liệu SymPy hoặc Python thành định dạng JSON-serializable.
    """
    # 1. Nếu là kiểu số nguyên thủy
    if isinstance(val, (int, float, bool, str)) or val is None:
        return val

    # 2. Nếu là cấu trúc tập hợp
    if isinstance(val, (list, tuple)):
        return [_format_value(item) for item in val]
    if isinstance(val, set):
        return [_format_value(item) for item in sorted(val, key=lambda x: str(x))]
    if isinstance(val, dict):
        return {str(k): _format_value(v) for k, v in val.items()}

    # 3. Kiểm tra kiểu SymPy nếu có
    try:
        import sympy
        if isinstance(val, (sympy.Integer, sympy.Rational, sympy.Float, sympy.Number)):
            f_val = float(val)
            if f_val.is_integer():
                return int(f_val)
            return f_val
        if isinstance(val, sympy.FiniteSet):
            return [_format_value(x) for x in val]
        if hasattr(val, "evalf"):
            try:
                f_val = float(val.evalf())
                if f_val.is_integer():
                    return int(f_val)
                return f_val
            except Exception:
                return str(val)
    except ImportError:
        pass

    return str(val)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Bộ nạp module an toàn chỉ cho phép import các module trong ALLOWED_MODULES.
    Tự động liên kết tới fallback_sympy nếu sympy chưa được cài đặt.
    """
    base_module = name.split(".")[0]
    if base_module not in ALLOWED_MODULES:
        raise ImportError(f"Quyền truy cập bị từ chối: Module '{name}' không được phép import trong sandbox.")

    try:
        return __import__(name, globals, locals, fromlist, level)
    except ModuleNotFoundError:
        if base_module == "sympy":
            import src.sandbox.fallback_sympy as fb_sympy
            return fb_sympy
        raise


def _build_safe_globals() -> Dict[str, Any]:
    """
    Tạo không gian tên toàn cục an toàn cho sandbox.
    """
    safe_builtins: Dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        "__import__": _safe_import,
    }

    safe_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
    }

    # Nạp math
    try:
        import math
        safe_globals["math"] = math
    except ImportError:
        pass

    # Nạp sympy nếu có, hoặc dùng fallback_sympy
    try:
        import sympy
        safe_globals["sympy"] = sympy
        for symbol_name in [
            "symbols", "Symbol", "Eq", "solve", "solveset", "linsolve", "nonlinsolve",
            "Rational", "S", "sqrt", "pi", "Abs", "Matrix", "simplify", "expand",
            "factor", "diff", "integrate", "Limit", "parse_expr", "oo"
        ]:
            if hasattr(sympy, symbol_name):
                safe_globals[symbol_name] = getattr(sympy, symbol_name)
    except ImportError:
        import src.sandbox.fallback_sympy as fb_sympy
        safe_globals["sympy"] = fb_sympy
        for symbol_name in [
            "symbols", "Symbol", "Eq", "solve", "solveset", "linsolve", "nonlinsolve",
            "Rational", "S", "sqrt", "pi", "Abs", "Matrix", "simplify", "expand",
            "factor", "oo"
        ]:
            if hasattr(fb_sympy, symbol_name):
                safe_globals[symbol_name] = getattr(fb_sympy, symbol_name)

    return safe_globals


def _worker_process_target(sanitized_code: str, queue: multiprocessing.Queue):
    """
    Worker chạy trong tiến trình con biệt lập để có thể force-kill khi timeout.
    """
    # Đăng ký fallback sympy nếu môi trường chưa cài đặt sympy
    try:
        import sympy
    except ImportError:
        import src.sandbox.fallback_sympy as fb_sympy
        sys.modules["sympy"] = fb_sympy

    safe_globals = _build_safe_globals()
    local_env: Dict[str, Any] = {}
    stdout_buffer = io.StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = stdout_buffer
        exec(sanitized_code, safe_globals, local_env)
        captured_stdout = stdout_buffer.getvalue().strip()

        # Tìm kiếm kết quả đầu ra
        result_value = None
        found_explicit = False
        for candidate_key in ["result", "solution", "ans", "final_answer", "res", "x_val"]:
            if candidate_key in local_env:
                result_value = _format_value(local_env[candidate_key])
                found_explicit = True
                break

        if not found_explicit:
            # Lấy biến tự do cuối cùng nếu có
            candidates = [v for k, v in local_env.items() if not k.startswith("_")]
            if candidates:
                result_value = _format_value(candidates[-1])
            elif captured_stdout:
                result_value = captured_stdout
            else:
                result_value = "Thực thi thành công (không có kết quả trả về)."

        queue.put((True, result_value, None, None))
    except Exception as exc:
        tb_str = traceback.format_exc()
        queue.put((False, None, f"{type(exc).__name__}: {str(exc)}", tb_str))
    finally:
        sys.stdout = original_stdout


class SafeSandboxExecutor:
    """
    Hộp cát thực thi mã Python/SymPy có phân tích AST và khống chế thời gian chờ.
    Sử dụng multiprocessing để đảm bảo ngắt tiến trình con ngay lập tức khi timeout.
    """

    def __init__(self, timeout: float = 3.0, max_ast_nodes: int = 1000):
        self.timeout = timeout
        self.max_ast_nodes = max_ast_nodes

    def execute(self, code: str) -> ExecutionResult:
        """
        Thực thi đoạn mã với kiểm tra tĩnh AST và quản lý timeout qua tiến trình độc lập.
        """
        start_time = time.time()

        # 1. Kiểm tra an ninh tĩnh bằng cây cú pháp trừu tượng AST
        is_safe, violation_reason, sanitized_code = validate_python_code(
            code, max_nodes=self.max_ast_nodes
        )
        if not is_safe:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"AST Security Violation: {violation_reason}",
                traceback=None,
                execution_time=time.time() - start_time,
                sanitized_code=sanitized_code,
                ast_safe=False,
                ast_violation=violation_reason
            )

        # 2. Khởi chạy tiến trình con biệt lập
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_worker_process_target,
            args=(sanitized_code, queue)
        )
        process.daemon = True
        process.start()

        process.join(timeout=self.timeout)

        duration = time.time() - start_time

        if process.is_alive():
            # Quá thời gian timeout -> chấm dứt tiến trình con ngay lập tức
            process.terminate()
            time.sleep(0.05)
            if process.is_alive():
                process.kill()
            process.join()

            return ExecutionResult(
                success=False,
                output=None,
                error=f"Execution Timeout: Thời gian thực thi vượt quá {self.timeout}s (vòng lặp vô hạn hoặc quá tải tính toán).",
                traceback="TimeoutError: Sandbox execution exceeded time limit.",
                execution_time=duration,
                sanitized_code=sanitized_code,
                ast_safe=True
            )

        if not queue.empty():
            success, output, err, tb = queue.get()
            return ExecutionResult(
                success=success,
                output=output,
                error=err,
                traceback=tb,
                execution_time=duration,
                sanitized_code=sanitized_code,
                ast_safe=True
            )

        # Trường hợp tiến trình bị crash đột ngột (OOM, segfault...)
        return ExecutionResult(
            success=False,
            output=None,
            error=f"Process Crashed: Tiến trình kết thúc bất thường với exitcode {process.exitcode}",
            traceback=None,
            execution_time=duration,
            sanitized_code=sanitized_code,
            ast_safe=True
        )

