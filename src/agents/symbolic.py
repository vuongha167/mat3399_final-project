# -*- coding: utf-8 -*-
"""
Tác tử Ký hiệu Toán học (Symbolic Agent).
Chuyển hóa thông tin phân tích thành hệ phương trình ký hiệu
và sinh mã Python sử dụng thư viện SymPy để giải chính xác.
"""

import json
from typing import Dict, Any, Optional
from src.llm.client import BaseLLMClient


SYMBOLIC_SYSTEM_PROMPT = """Bạn là chuyên gia Đại số Ký hiệu (Symbolic Agent) chuyên về thư viện SymPy trong Python.
Nhiệm vụ của bạn là tiếp nhận bài toán THCS và thông tin phân tích cấu trúc, sau đó sinh mã Python sử dụng thư viện `sympy` để giải nghiệm chính xác tuyệt đối.

CÁC NGUYÊN TẮC BẮT BUỘC:
1. Chỉ sử dụng thư viện `sympy` và `math`. CẤM TUYỆT ĐỐI import các module hệ thống (os, sys, subprocess, open, eval...).
2. Khai báo các ký hiệu bằng `symbols('x y ...', real=True, positive=True)` hoặc tương ứng với điều kiện ẩn số.
3. Sử dụng `Eq(vế_trái, vế_phải)` để biểu diễn các phương trình.
4. Với phân số, sử dụng `Rational(tử, mẫu)` hoặc `S(1)/2` để tránh sai số dấu phẩy động (floating point).
5. Sau khi giải bằng `solve(...)`, hãy lọc nghiệm thỏa mãn điều kiện đề bài (ví dụ loại nghiệm âm khi vận tốc > 0, loại nghiệm không nguyên nếu là số người/sản phẩm).
6. Kết quả cuối cùng BẮT BUỘC phải gán vào biến `result` (có thể là số, danh sách hoặc từ điển chứa giá trị các ẩn).
7. Xuất DUY NHẤT một khối mã nguồn Python đặt trong cặp dấu ```python ... ```, không viết thêm lời dẫn dài dòng.

Mẫu mã SymPy chuẩn tham khảo:
```python
from sympy import symbols, Eq, solve, Rational

# 1. Khai báo ẩn số với điều kiện
x = symbols('x', positive=True, real=True)

# 2. Thiết lập phương trình theo đề bài
# Ví dụ: 120 / x - 120 / (x + 10) = Rational(1, 2)
eq = Eq(120 / x - 120 / (x + 10), Rational(1, 2))

# 3. Giải phương trình
solutions = solve(eq, x)

# 4. Lọc nghiệm thỏa mãn điều kiện
valid_solutions = [sol for sol in solutions if sol > 0]

# 5. Gán kết quả cuối cùng
result = valid_solutions[0] if valid_solutions else None
```
"""


class SymbolicAgent:
    """Tác tử sinh mã đại số ký hiệu SymPy."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def generate_code(
        self,
        problem_text: str,
        parsed_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Sinh mã Python SymPy từ đề bài và thông tin cấu trúc.
        """
        parsed_str = json.dumps(parsed_info or {}, ensure_ascii=False, indent=2)

        user_prompt = f"""Đề bài toán:
{problem_text}

Thông tin phân tích cấu trúc từ Parser Agent:
{parsed_str}

Hãy xây dựng phương trình/hệ phương trình và sinh mã Python SymPy hoàn chỉnh để tìm ra đáp số."""

        response = self.llm.generate(
            prompt=user_prompt,
            system_instruction=SYMBOLIC_SYSTEM_PROMPT,
            temperature=0.1
        )

        return response.strip()
