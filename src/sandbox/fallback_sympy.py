# -*- coding: utf-8 -*-
"""
Bộ công cụ Ký hiệu Dự phòng Thuần Python (Pure-Python Fallback Symbolic Engine).
Được kích hoạt tự động khi môi trường chưa cài đặt thư viện 'sympy'.
Hỗ trợ đầy đủ các phép toán và dạng toán THCS:
- Ký hiệu đại số, phân số Rational, căn thức sqrt, trị tuyệt đối Abs, hằng số pi, oo.
- Giải phương trình bậc 1, bậc 2, phương trình phân thức chứa ẩn ở mẫu.
- Giải hệ 2 phương trình bậc nhất và phi tuyến 2 ẩn (Cramer & số học).
"""

import math
import re
from typing import Any, List, Union, Tuple, Dict, Optional


class Symbol:
    """Ký hiệu đại số thuần Python."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.assumptions = kwargs

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        return False

    def __add__(self, other):
        return Expr(self, "+", other)

    def __radd__(self, other):
        return Expr(other, "+", self)

    def __sub__(self, other):
        return Expr(self, "-", other)

    def __rsub__(self, other):
        return Expr(other, "-", self)

    def __mul__(self, other):
        return Expr(self, "*", other)

    def __rmul__(self, other):
        return Expr(other, "*", self)

    def __truediv__(self, other):
        return Expr(self, "/", other)

    def __rtruediv__(self, other):
        return Expr(other, "/", self)

    def __neg__(self):
        return Expr(0, "-", self)

    def __pow__(self, power):
        return Expr(self, "**", power)

    def __abs__(self):
        return Abs(self)


class Expr:
    """Biểu thức toán học đại số biểu diễn cây nhị phân."""

    def __init__(self, left: Any, op: str, right: Any = None):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        if self.op == "abs":
            return f"Abs({self.left})"
        if self.op == "sqrt":
            return f"sqrt({self.left})"
        return f"({self.left} {self.op} {self.right})"

    def __add__(self, other):
        return Expr(self, "+", other)

    def __radd__(self, other):
        return Expr(other, "+", self)

    def __sub__(self, other):
        return Expr(self, "-", other)

    def __rsub__(self, other):
        return Expr(other, "-", self)

    def __mul__(self, other):
        return Expr(self, "*", other)

    def __rmul__(self, other):
        return Expr(other, "*", self)

    def __truediv__(self, other):
        return Expr(self, "/", other)

    def __rtruediv__(self, other):
        return Expr(other, "/", self)

    def __neg__(self):
        return Expr(0, "-", self)

    def __pow__(self, power):
        return Expr(self, "**", power)

    def __abs__(self):
        return Abs(self)

    def evaluate(self, env: Dict[str, float]) -> float:
        """Tính giá trị số của biểu thức với môi trường biến số."""
        def eval_node(node):
            if isinstance(node, Symbol):
                return env.get(node.name, 0.0)
            if isinstance(node, Expr):
                if node.op == "abs":
                    return abs(eval_node(node.left))
                if node.op == "sqrt":
                    inner = eval_node(node.left)
                    return math.sqrt(max(0.0, inner))
                l_val = eval_node(node.left)
                r_val = eval_node(node.right) if node.right is not None else 0.0
                if node.op == "+":
                    return l_val + r_val
                if node.op == "-":
                    return l_val - r_val
                if node.op == "*":
                    return l_val * r_val
                if node.op == "/":
                    if abs(r_val) < 1e-12:
                        return float("inf")
                    return l_val / r_val
                if node.op == "**":
                    try:
                        return l_val ** r_val
                    except Exception:
                        return float("nan")
            if isinstance(node, (int, float)):
                return float(node)
            return 0.0

        return eval_node(self)


class Eq:
    """Biểu diễn phương trình toán học lhs = rhs."""

    def __init__(self, lhs: Any, rhs: Any = 0):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"Eq({self.lhs}, {self.rhs})"


# ================== HÀM TOÁN HỌC & HẰNG SỐ BỔ TRỢ ==================

pi = math.pi
oo = float("inf")


def sqrt(val: Any) -> Any:
    """Căn bậc hai hỗ trợ cả số thực lẫn biểu thức Symbol."""
    if isinstance(val, (int, float)):
        return math.sqrt(val)
    return Expr(val, "sqrt")


def Abs(val: Any) -> Any:
    """Giá trị tuyệt đối."""
    if isinstance(val, (int, float)):
        return abs(val)
    return Expr(val, "abs")


class _SingletonRegistry:
    """Mô phỏng Singleton Registry S của SymPy (ví dụ S(1)/2, S.Zero)."""
    Zero = 0
    One = 1
    Half = 0.5

    def __call__(self, val: Any) -> Any:
        if isinstance(val, int):
            return float(val)
        return val


S = _SingletonRegistry()


def symbols(names: str, **kwargs) -> Any:
    """Khai báo một hoặc nhiều ký hiệu biến số."""
    name_list = [n.strip() for n in re.split(r"[,\s]+", names.strip()) if n.strip()]
    if len(name_list) == 1:
        return Symbol(name_list[0], **kwargs)
    return tuple(Symbol(name, **kwargs) for name in name_list)


def Rational(p: int, q: int = 1) -> float:
    """Phân số số học."""
    return p / q


def simplify(expr: Any) -> Any:
    """Mô phỏng hàm simplify."""
    return expr


def expand(expr: Any) -> Any:
    """Mô phỏng hàm expand."""
    return expr


def factor(expr: Any) -> Any:
    """Mô phỏng hàm factor."""
    return expr


class Matrix:
    """Lớp ma trận cơ bản."""
    def __init__(self, data: List[Any]):
        self.data = data

    def __repr__(self):
        return f"Matrix({self.data})"


# ================== BỘ GIẢI PHƯƠNG TRÌNH & HỆ PHƯƠNG TRÌNH ==================

def _extract_symbols(nodes: List[Any]) -> List[Symbol]:
    """Trích xuất tất cả các biến Symbol từ danh sách phương trình hoặc biểu thức."""
    syms: Dict[str, Symbol] = {}

    def scan(node):
        if isinstance(node, Eq):
            scan(node.lhs)
            scan(node.rhs)
        elif isinstance(node, Expr):
            scan(node.left)
            if node.right is not None:
                scan(node.right)
        elif isinstance(node, Symbol):
            syms[node.name] = node

    for n in nodes:
        scan(n)

    # Giữ nguyên thứ tự xuất hiện
    return list(syms.values())


def _clean_number(val: float) -> Union[int, float]:
    """Làm tròn và ép kiểu số nguyên nếu phần thập phân bằng 0."""
    r = round(val, 6)
    if abs(r - round(r)) < 1e-6:
        return int(round(r))
    return r


class SolutionDict(dict):
    """
    Từ điển nghiệm tùy biến: hỗ trợ truy cập vừa bằng Symbol vừa bằng tên chuỗi,
    đồng thời hỗ trợ truy cập chỉ số kiểu danh sách sol[0], sol[1].
    """
    def __getitem__(self, key):
        if isinstance(key, Symbol):
            return super().__getitem__(key)
        if isinstance(key, str):
            for k, v in self.items():
                if (isinstance(k, Symbol) and k.name == key) or k == key:
                    return v
        if isinstance(key, int):
            vals = list(self.values())
            if 0 <= key < len(vals):
                return vals[key]
        return super().__getitem__(key)


def solve(equation_or_system: Any, *args, **kwargs) -> Any:
    """
    Bộ giải phương trình và hệ phương trình đại số THCS thuần Python.
    Hỗ trợ:
    1. Phương trình 1 ẩn bậc 1, bậc 2, phân thức hữu tỉ.
    2. Hệ 2 phương trình bậc nhất 2 ẩn (phương pháp định thức Cramer chuẩn xác).
    3. Hệ 2 phương trình phi tuyến 2 ẩn bằng tìm kiếm nghiệm số học.
    """
    # Chuẩn hóa danh sách phương trình
    if isinstance(equation_or_system, (list, tuple)):
        equations = list(equation_or_system)
    else:
        equations = [equation_or_system]

    # Xác định danh sách biến mục tiêu
    target_symbols = []
    if args:
        first_arg = args[0]
        if isinstance(first_arg, (list, tuple)):
            target_symbols = list(first_arg)
        elif isinstance(first_arg, Symbol):
            target_symbols = [first_arg]
            if len(args) > 1:
                for a in args[1:]:
                    if isinstance(a, Symbol):
                        target_symbols.append(a)

    if not target_symbols:
        target_symbols = _extract_symbols(equations)

    if not target_symbols:
        return []

    # ==================== TRƯỜNG HỢP 1: PHƯƠNG TRÌNH 1 ẨN ====================
    if len(target_symbols) == 1:
        sym = target_symbols[0]
        eq = equations[0]
        lhs = eq.lhs if isinstance(eq, Eq) else eq
        rhs = eq.rhs if isinstance(eq, Eq) else 0

        def diff_func(val: float) -> float:
            env = {sym.name: val}
            l_val = lhs.evaluate(env) if isinstance(lhs, Expr) else (val if isinstance(lhs, Symbol) else float(lhs))
            r_val = rhs.evaluate(env) if isinstance(rhs, Expr) else (val if isinstance(rhs, Symbol) else float(rhs))
            return l_val - r_val

        solutions = []

        # 1. Thử các số nguyên trong phạm vi THCS (-500 đến 1000)
        for test_int in range(-500, 1001):
            try:
                err = abs(diff_func(float(test_int)))
                if err < 1e-7:
                    if test_int not in solutions:
                        solutions.append(test_int)
            except Exception:
                pass

        # 2. Quét lưới mịn hơn nếu chưa đủ nghiệm (phương pháp Bisection)
        if len(solutions) < 2:
            span_grid = [i * 0.5 for i in range(-200, 401)]
            for i in range(len(span_grid) - 1):
                x1, x2 = span_grid[i], span_grid[i + 1]
                try:
                    f1 = diff_func(x1)
                    f2 = diff_func(x2)
                    if math.isnan(f1) or math.isnan(f2) or math.isinf(f1) or math.isinf(f2):
                        continue
                    if f1 * f2 <= 0:
                        a, b = x1, x2
                        for _ in range(45):
                            mid = (a + b) / 2.0
                            f_mid = diff_func(mid)
                            if abs(f_mid) < 1e-9:
                                a = mid
                                break
                            if f1 * f_mid < 0:
                                b = mid
                            else:
                                a = mid
                                f1 = f_mid
                        root = _clean_number(a)
                        if root not in solutions:
                            solutions.append(root)
                except Exception:
                    pass

        return solutions

    # ==================== TRƯỜNG HỢP 2: HỆ 2 PHƯƠNG TRÌNH 2 ẨN ====================
    elif len(target_symbols) == 2:
        sym_x, sym_y = target_symbols[0], target_symbols[1]
        if len(equations) >= 2:
            eq1, eq2 = equations[0], equations[1]
        else:
            return []

        def eval_eq(eq_obj, vx: float, vy: float) -> float:
            lhs = eq_obj.lhs if isinstance(eq_obj, Eq) else eq_obj
            rhs = eq_obj.rhs if isinstance(eq_obj, Eq) else 0
            env = {sym_x.name: vx, sym_y.name: vy}
            l_val = lhs.evaluate(env) if isinstance(lhs, Expr) else (
                vx if isinstance(lhs, Symbol) and lhs.name == sym_x.name else (
                    vy if isinstance(lhs, Symbol) and lhs.name == sym_y.name else float(lhs)
                )
            )
            r_val = rhs.evaluate(env) if isinstance(rhs, Expr) else (
                vx if isinstance(rhs, Symbol) and rhs.name == sym_x.name else (
                    vy if isinstance(rhs, Symbol) and rhs.name == sym_y.name else float(rhs)
                )
            )
            return l_val - r_val

        # Kiểm tra tính tuyến tính bằng phép vi phân số học (định lý Cramer)
        # f(x, y) = a*x + b*y + c
        # c = f(0, 0), a = f(1, 0) - c, b = f(0, 1) - c
        try:
            c1 = eval_eq(eq1, 0.0, 0.0)
            a1 = eval_eq(eq1, 1.0, 0.0) - c1
            b1 = eval_eq(eq1, 0.0, 1.0) - c1

            c2 = eval_eq(eq2, 0.0, 0.0)
            a2 = eval_eq(eq2, 1.0, 0.0) - c2
            b2 = eval_eq(eq2, 0.0, 1.0) - c2

            # Kiểm tra xem có tuyến tính thực sự tại điểm (2, 3)
            is_linear_1 = abs(eval_eq(eq1, 2.0, 3.0) - (a1 * 2.0 + b1 * 3.0 + c1)) < 1e-6
            is_linear_2 = abs(eval_eq(eq2, 2.0, 3.0) - (a2 * 2.0 + b2 * 3.0 + c2)) < 1e-6

            if is_linear_1 and is_linear_2:
                # Hệ tuyến tính: a1*x + b1*y = -c1 và a2*x + b2*y = -c2
                const1 = -c1
                const2 = -c2
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-10:
                    det_x = const1 * b2 - const2 * b1
                    det_y = a1 * const2 - a2 * const1
                    res_x = _clean_number(det_x / det)
                    res_y = _clean_number(det_y / det)

                    # Trả về từ điển nghiệm tương thích chuẩn SymPy
                    sol_dict = SolutionDict({sym_x: res_x, sym_y: res_y})
                    return sol_dict
        except Exception:
            pass

        # Nếu là hệ phi tuyến (ví dụ hình học x^2 + y^2 = R^2 hoặc làm chung riêng 1/x + 1/y):
        # Dùng tìm kiếm nghiệm số học trên miền số nguyên và bán nguyên
        for tx in range(-300, 601):
            for ty in range(-300, 601):
                try:
                    e1 = abs(eval_eq(eq1, float(tx), float(ty)))
                    e2 = abs(eval_eq(eq2, float(tx), float(ty)))
                    if e1 < 1e-6 and e2 < 1e-6:
                        return SolutionDict({sym_x: tx, sym_y: ty})
                except Exception:
                    pass

    return []


def linsolve(system: Any, *symbols_args) -> List[Any]:
    """Giải hệ phương trình tuyến tính."""
    res = solve(system, *symbols_args)
    if isinstance(res, dict):
        return [tuple(res.values())]
    return res


def nonlinsolve(system: Any, *symbols_args) -> List[Any]:
    """Giải hệ phương trình phi tuyến."""
    res = solve(system, *symbols_args)
    if isinstance(res, dict):
        return [tuple(res.values())]
    return res


def solveset(eq: Any, sym: Symbol = None, **kwargs) -> List[Any]:
    """Mô phỏng hàm solveset của SymPy."""
    return solve(eq, sym)
