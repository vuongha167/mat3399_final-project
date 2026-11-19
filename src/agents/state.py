# -*- coding: utf-8 -*-
"""
Định nghĩa cấu trúc trạng thái chung (AgentState) cho hệ thống Multi-Agent LangGraph.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Cấu trúc trạng thái luồng điều phối Multi-Agent.
    """
    problem: str                                # Văn bản đề bài toán ban đầu
    parsed_info: Optional[Dict[str, Any]]       # Thông tin trích xuất: ẩn số, điều kiện, giả thiết
    sympy_code: Optional[str]                   # Mã Python SymPy được sinh ra
    execution_result: Optional[Dict[str, Any]]  # Kết quả chạy trong Sandbox
    correction_history: List[Dict[str, Any]]    # Lịch sử các lần tự sửa lỗi
    attempt_count: int                          # Số lần đã thử sửa mã (0, 1, 2, ...)
    max_attempts: int                           # Giới hạn số lần sửa mã tối đa (3)
    final_solution: Optional[str]               # Lời giải sư phạm chuẩn SGK kèm LaTeX
    status: str                                 # Trạng thái hiện tại của luồng
    error_log: Optional[str]                    # Nhật ký lỗi nếu xảy ra sự cố
    agent_logs: List[Dict[str, Any]]            # Nhật ký chi tiết của từng Agent phục vụ hiển thị Streamlit
