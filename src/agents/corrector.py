# -*- coding: utf-8 -*-
"""
Tác tử Tự Sửa Lỗi (Self-Correction Agent).
Phân tích nguyên nhân khi mã SymPy thực thi thất bại trong Sandbox
(lỗi cú pháp, ngoại lệ thời gian chạy, vi phạm AST, kết quả rỗng, timeout)
và tự động tái cấu trúc, sinh lại mã Python sửa chữa chính xác.
"""

from typing import Dict, Any, Optional
from src.llm.client import BaseLLMClient


CORRECTOR_SYSTEM_PROMPT = """Bạn là chuyên gia Phục hồi & Tự sửa lỗi Mã nguồn (Self-Correction Agent) cho hệ thống giải toán SymPy.
Nhiệm vụ của bạn là kiểm tra mã nguồn bị lỗi, đọc kỹ thông báo lỗi / vết ngăn xếp ngoại lệ (traceback), tìm ra nguyên nhân gốc rễ và sửa lại đoạn mã để thực thi thành công trong Sandbox.

CÁC NGUYÊN TẮC QUAN TRỌNG:
1. Đọc kỹ thông báo lỗi và dòng mã gây ra sự cố.
2. CÁC NGUYÊN NHÂN LỖI THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC:
   - "NameError: name 'x' is not defined": Cần khai báo ký hiệu bằng `x = symbols('x')`.
   - "TypeError: unsupported operand": Cần dùng `Rational(a, b)` thay vì phân số chưa định dạng hoặc kiểm tra phép cộng biểu thức.
   - "Empty solution list []": Phương trình thiết lập bị mâu thuẫn hoặc chưa lọc đúng điều kiện. Hãy kiểm tra lại logic đề bài.
   - "AST Security Violation": Mã chứa module hoặc hàm bị cấm (như os, sys, eval, open...). Hãy loại bỏ hoàn toàn các lệnh này, chỉ dùng `sympy` và `math`.
   - "Timeout": Vòng lặp quá dài hoặc giải phương trình phi tuyến tính quá nặng; hãy đơn giản hóa bằng cách đặt ẩn phụ hoặc dùng `linsolve` / `solve`.
3. Đoạn mã mới BẮT BUỘC phải gán đáp án cuối cùng vào biến `result`.
4. Xuất DUY NHẤT một khối mã Python trong ```python ... ```. Không thêm văn bản giải thích dài dòng ngoài khối mã.
"""


class SelfCorrectionAgent:
    """Tác tử tự sửa mã SymPy bị lỗi."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def correct_code(
        self,
        problem_text: str,
        failed_code: str,
        error_message: str,
        traceback_str: Optional[str] = None,
        attempt: int = 1,
        parsed_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Phân tích lỗi và sinh mã sửa chữa.
        """
        user_prompt = f"""ĐỀ BÀI TOÁN:
{problem_text}

MÃ NGUỒN BỊ LỖI (Lần thử {attempt}):
```python
{failed_code}
```

THÔNG BÁO LỖI TỪ SANDBOX:
{error_message}

TRACEBACK CHI TIẾT:
{traceback_str or "Không có traceback chi tiết."}

Hãy phân tích lỗi trên, sửa lại toàn bộ mã nguồn Python/SymPy sao cho chạy thành công và lưu kết quả vào biến `result`."""

        response = self.llm.generate(
            prompt=user_prompt,
            system_instruction=CORRECTOR_SYSTEM_PROMPT,
            temperature=0.1
        )

        return response.strip()
