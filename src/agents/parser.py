# -*- coding: utf-8 -*-
"""
Tác tử Phân tích Đề bài (Parser Agent).
Đọc hiểu đề bài toán THCS tiếng Việt, bóc tách cấu trúc toán học:
loại bài toán, các biến/ẩn số, điều kiện thực tế, giả thiết và câu hỏi cần giải.
"""

import json
import re
from typing import Dict, Any, Optional
from src.llm.client import BaseLLMClient


PARSER_SYSTEM_PROMPT = """Bạn là chuyên gia Phân tích Đề bài Toán THCS (Parser Agent) thuộc hệ thống AI Tutor.
Nhiệm vụ của bạn là tiếp nhận đề bài toán THCS tiếng Việt và bóc tách cấu trúc toán học một cách chính xác, logic và đầy đủ.

Hãy phân tích và trả về DUY NHẤT một khối JSON hợp lệ theo định dạng sau (không viết thêm lời dẫn nào bên ngoài):
```json
{
  "problem_type": "Loại bài toán (ví dụ: Toán chuyển động, Toán năng suất / công việc chung riêng, Toán nồng độ dung dịch, Toán phần trăm, Toán hình học thực tế, Giải hệ phương trình...)",
  "entities": ["Danh sách các đối tượng tham gia: xe máy, ô tô, hai vòi nước..."],
  "variables": [
    {
      "name": "x",
      "meaning": "Vận tốc xe máy",
      "unit": "km/h",
      "condition": "x > 0"
    }
  ],
  "givens": [
    "Quãng đường AB dài 120 km",
    "Xe máy đi trước ô tô 30 phút (0.5 giờ)",
    "Vận tốc ô tô lớn hơn xe máy 10 km/h"
  ],
  "relationships": [
    "Thời gian xe máy = 120 / x",
    "Thời gian ô tô = 120 / (x + 10)",
    "Thời gian xe máy - Thời gian ô tô = 0.5"
  ],
  "target": "Tìm vận tốc của mỗi xe"
}
```

Lưu ý:
1. Đảm bảo điều kiện của ẩn số phải phù hợp thực tế (vận tốc > 0, số người/sản phẩm nguyên dương...).
2. Đổi các đại lượng thời gian về cùng một đơn vị (ví dụ phút sang giờ: 30 phút = 1/2 giờ).
3. Đảm bảo JSON hợp lệ, không chứa ký tự escape bị lỗi.
"""


class ParserAgent:
    """Tác tử phân tích đề bài toán THCS."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def parse(self, problem_text: str) -> Dict[str, Any]:
        """
        Phân tích đề bài toán và trả về từ điển thông tin cấu trúc.
        """
        user_prompt = f"Phân tích đề bài toán THCS sau đây:\n\n{problem_text}"
        raw_response = self.llm.generate(
            prompt=user_prompt,
            system_instruction=PARSER_SYSTEM_PROMPT,
            temperature=0.1
        )

        return self._extract_json(raw_response, problem_text)

    def _extract_json(self, response_text: str, original_problem: str) -> Dict[str, Any]:
        """Trích xuất và kiểm tra tính hợp lệ của JSON từ phản hồi."""
        # Thử tìm khối ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text, re.IGNORECASE)
        json_str = json_match.group(1).strip() if json_match else response_text.strip()

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback tạo cấu trúc mặc định nếu LLM không trả về JSON chuẩn
        return {
            "problem_type": "Toán đại số / Suy luận số đa bước",
            "entities": ["Toán học THCS"],
            "variables": [
                {"name": "x", "meaning": "Ẩn số chính", "unit": "", "condition": "x thuộc R"}
            ],
            "givens": [original_problem],
            "relationships": ["Theo đề bài thiết lập phương trình"],
            "target": "Tìm giá trị thỏa mãn bài toán",
            "raw_response": response_text
        }
