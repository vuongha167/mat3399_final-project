# -*- coding: utf-8 -*-
"""
Tác tử Sư phạm Toán THCS (Pedagogical Agent).
Tiếp nhận nghiệm số học từ Sandbox và thông tin phân tích để xây dựng
lời giải sư phạm chuẩn 5 bước theo sách giáo khoa Toán THCS Việt Nam,
trình bày công thức toán học bằng định dạng LaTeX chuẩn đẹp.
"""

import json
from typing import Dict, Any, Optional
from src.llm.client import BaseLLMClient


PEDAGOGICAL_SYSTEM_PROMPT = """Bạn là Giáo viên Dạy giỏi Toán THCS kiêm Tác tử Sư phạm (Pedagogical Agent) thuộc hệ thống AI Tutor.
Nhiệm vụ của bạn là trình bày lời giải bài toán theo đúng phương pháp sư phạm chuẩn sách giáo khoa (SGK) Toán THCS Việt Nam (Chương trình GDPT 2018).

CẤU TRÚC LỜI GIẢI 5 BƯỚC BẮT BUỘC:
1. **Bước 1: Chọn ẩn số và đặt điều kiện thích hợp cho ẩn.** (Nêu rõ ẩn số đại diện cho cái gì, đơn vị đo, và điều kiện toán học cũng như thực tế).
2. **Bước 2: Biểu diễn các đại lượng chưa biết theo ẩn và các đại lượng đã biết.** (Giải thích rõ ràng mối quan hệ giữa các đại lượng).
3. **Bước 3: Lập phương trình (hoặc hệ phương trình).** (Nêu căn cứ từ đề bài để lập nên phương trình).
4. **Bước 4: Giải phương trình (hệ phương trình) từng bước.** (Biến đổi tương đương, quy đồng, rút gọn, tìm nghiệm rõ ràng).
5. **Bước 5: Đối chiếu nghiệm với điều kiện ban đầu và kết luận bài toán.** (Chỉ rõ nghiệm nào thỏa mãn, nghiệm nào bị loại và trả lời câu hỏi của đề bài).

QUY TẮC ĐỊNH DẠNG:
- Sử dụng mã LaTeX chuẩn cho tất cả công thức toán học:
  + Công thức trong dòng dùng cặp dấu `$ ... $`, ví dụ: `$x > 0$`, `$\\frac{120}{x}$`.
  + Công thức dòng riêng (khối) dùng cặp dấu `$$ ... $$`, ví dụ:
    $$ \\frac{120}{x} - \\frac{120}{x+10} = \\frac{1}{2} $$
- Sử dụng bảng hoặc hệ phương trình `\\begin{cases} ... \\end{cases}` khi giải hệ.
- Giọng văn mô phạm, chuẩn mực, dễ hiểu đối với học sinh lớp 6, 7, 8, 9.
- Đáp số cuối cùng BẮT BUỘC phải khớp hoàn toàn với nghiệm đã được kiểm chứng bởi Sandbox.
"""


class PedagogicalAgent:
    """Tác tử diễn giải sư phạm chuẩn SGK Toán THCS."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def explain(
        self,
        problem_text: str,
        parsed_info: Optional[Dict[str, Any]],
        symbolic_result: Any,
        sympy_code: Optional[str] = None
    ) -> str:
        """
        Sinh lời giải sư phạm chi tiết kèm công thức LaTeX.
        """
        parsed_str = json.dumps(parsed_info or {}, ensure_ascii=False, indent=2)

        user_prompt = f"""ĐỀ BÀI TOÁN THCS:
{problem_text}

CẤU TRÚC PHÂN TÍCH (Parser Agent):
{parsed_str}

KẾT QUẢ ĐÃ ĐƯỢC HỘP CÁT SYMPY TÍNH TOÁN XÁC THỰC:
{symbolic_result}

MÃ SYMPY THỰC THI (nếu có):
```python
{sympy_code or "# Không có mã nguồn"}
```

Hãy soạn thảo lời giải hoàn chỉnh theo đúng chuẩn 5 bước sư phạm SGK Toán THCS, dùng định dạng LaTeX chuẩn đẹp."""

        response = self.llm.generate(
            prompt=user_prompt,
            system_instruction=PEDAGOGICAL_SYSTEM_PROMPT,
            temperature=0.2
        )

        return response.strip()
