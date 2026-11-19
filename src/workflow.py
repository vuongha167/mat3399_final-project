# -*- coding: utf-8 -*-
"""
Bộ điều phối quy trình Multi-Agent (Workflow Orchestrator).
Quản lý luồng trạng thái tuần hoàn giữa Parser -> Symbolic -> Sandbox -> Self-Correction -> Pedagogical.
Hỗ trợ cả LangGraph StateGraph và Pure Python Runner (đảm bảo khả năng chạy độc lập 100%).
"""

import time
from typing import Dict, Any, Optional, Callable, Generator
import config
from src.agents.state import AgentState
from src.agents.parser import ParserAgent
from src.agents.symbolic import SymbolicAgent
from src.agents.corrector import SelfCorrectionAgent
from src.agents.pedagogical import PedagogicalAgent
from src.sandbox.executor import SafeSandboxExecutor
from src.llm.client import BaseLLMClient


class MultiAgentMathSolver:
    """
    Hệ thống điều phối Multi-Agent kết hợp kiểm chứng ký hiệu SymPy Sandbox.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        timeout: float = float(config.SANDBOX_TIMEOUT_SECONDS),
        max_attempts: int = config.MAX_CORRECTION_ATTEMPTS
    ):
        self.llm = llm_client
        self.timeout = timeout
        self.max_attempts = max_attempts

        # Khởi tạo các tác tử
        self.parser_agent = ParserAgent(self.llm)
        self.symbolic_agent = SymbolicAgent(self.llm)
        self.corrector_agent = SelfCorrectionAgent(self.llm)
        self.pedagogical_agent = PedagogicalAgent(self.llm)
        self.sandbox = SafeSandboxExecutor(timeout=self.timeout)

    # ================== CÁC NÚT (NODES) TRONG ĐỒ THỊ ==================

    def parser_node(self, state: AgentState) -> AgentState:
        """Nút 1: Phân tích đề bài toán."""
        t0 = time.time()
        parsed = self.parser_agent.parse(state["problem"])
        state["parsed_info"] = parsed
        state["status"] = "parsed"
        state["agent_logs"].append({
            "agent": "Parser Agent",
            "step": "Bóc tách cấu trúc đề bài",
            "duration": time.time() - t0,
            "data": parsed
        })
        return state

    def symbolic_node(self, state: AgentState) -> AgentState:
        """Nút 2: Xây dựng hệ thức ký hiệu và sinh mã SymPy."""
        t0 = time.time()
        code = self.symbolic_agent.generate_code(
            state["problem"],
            state.get("parsed_info")
        )
        state["sympy_code"] = code
        state["status"] = "symbolic_generated"
        state["agent_logs"].append({
            "agent": "Symbolic Agent",
            "step": "Sinh mã SymPy ký hiệu",
            "duration": time.time() - t0,
            "data": code
        })
        return state

    def executor_node(self, state: AgentState) -> AgentState:
        """Nút 3: Thực thi mã trong Safe Sandbox có kiểm soát AST và Timeout."""
        t0 = time.time()
        code_to_run = state.get("sympy_code", "")
        exec_res = self.sandbox.execute(code_to_run)

        res_dict = {
            "success": exec_res.success,
            "output": exec_res.output,
            "error": exec_res.error,
            "traceback": exec_res.traceback,
            "execution_time": exec_res.execution_time,
            "sanitized_code": exec_res.sanitized_code,
            "ast_safe": exec_res.ast_safe,
            "ast_violation": exec_res.ast_violation
        }
        state["execution_result"] = res_dict

        # Đánh giá kết quả: nếu output rỗng hoặc None thì cũng coi như chưa thành công
        if exec_res.success:
            if exec_res.output is None or exec_res.output == [] or exec_res.output == "[]":
                res_dict["success"] = False
                res_dict["error"] = "Tập nghiệm rỗng (không tìm thấy nghiệm thỏa mãn phương trình)."

        state["agent_logs"].append({
            "agent": "Sandbox Executor",
            "step": f"Thực thi Sandbox (Lần {state.get('attempt_count', 0) + 1})",
            "duration": time.time() - t0,
            "data": res_dict
        })
        return state

    def corrector_node(self, state: AgentState) -> AgentState:
        """Nút 4: Tự sửa lỗi mã khi Sandbox phát hiện ngoại lệ hoặc nghiệm rỗng."""
        t0 = time.time()
        current_attempts = state.get("attempt_count", 0) + 1
        state["attempt_count"] = current_attempts

        last_res = state.get("execution_result", {})
        err_msg = last_res.get("error", "Lỗi không xác định")
        tb_str = last_res.get("traceback")
        code = state.get("sympy_code", "")

        state["correction_history"].append({
            "attempt": current_attempts,
            "failed_code": code,
            "error": err_msg,
            "traceback": tb_str
        })

        corrected_code = self.corrector_agent.correct_code(
            problem_text=state["problem"],
            failed_code=code,
            error_message=err_msg,
            traceback_str=tb_str,
            attempt=current_attempts,
            parsed_info=state.get("parsed_info")
        )

        state["sympy_code"] = corrected_code
        state["status"] = "corrected"
        state["agent_logs"].append({
            "agent": "Self-Correction Agent",
            "step": f"Tự động sửa lỗi mã (Lần {current_attempts})",
            "duration": time.time() - t0,
            "data": {
                "attempt": current_attempts,
                "error_resolved": err_msg,
                "corrected_code": corrected_code
            }
        })
        return state

    def pedagogical_node(self, state: AgentState) -> AgentState:
        """Nút 5: Tổng hợp và soạn thảo lời giải sư phạm chuẩn SGK THCS kèm LaTeX."""
        t0 = time.time()
        last_res = state.get("execution_result", {})
        output_val = last_res.get("output", "Không xác định")

        explanation = self.pedagogical_agent.explain(
            problem_text=state["problem"],
            parsed_info=state.get("parsed_info"),
            symbolic_result=output_val,
            sympy_code=state.get("sympy_code")
        )

        state["final_solution"] = explanation
        state["status"] = "completed"
        state["agent_logs"].append({
            "agent": "Pedagogical Agent",
            "step": "Trình bày sư phạm chuẩn SGK THCS",
            "duration": time.time() - t0,
            "data": explanation
        })
        return state

    # ================== BỘ CHẠY QUY TRÌNH (RUNNERS) ==================

    def run(self, problem_text: str, prefer_langgraph: bool = True) -> AgentState:
        """
        Thực thi toàn bộ luồng quy trình giải bài toán tuần hoàn.
        Ưu tiên điều phối qua LangGraph StateGraph nếu thư viện langgraph có sẵn trong môi trường,
        tự động fallback sang Pure-Python runner nếu chưa cài đặt.
        """
        if prefer_langgraph:
            langgraph_app = build_langgraph_workflow(
                self.llm, timeout=self.timeout, max_attempts=self.max_attempts
            )
            if langgraph_app is not None:
                initial_state: AgentState = {
                    "problem": problem_text.strip(),
                    "parsed_info": None,
                    "sympy_code": None,
                    "execution_result": None,
                    "correction_history": [],
                    "attempt_count": 0,
                    "max_attempts": self.max_attempts,
                    "final_solution": None,
                    "status": "start",
                    "error_log": None,
                    "agent_logs": []
                }
                return langgraph_app.invoke(initial_state)

        state: AgentState = {
            "problem": problem_text.strip(),
            "parsed_info": None,
            "sympy_code": None,
            "execution_result": None,
            "correction_history": [],
            "attempt_count": 0,
            "max_attempts": self.max_attempts,
            "final_solution": None,
            "status": "start",
            "error_log": None,
            "agent_logs": []
        }

        # 1. Parser Node
        state = self.parser_node(state)

        # 2. Symbolic Node
        state = self.symbolic_node(state)

        # 3. Vòng lặp Executor <-> Corrector
        while True:
            state = self.executor_node(state)
            exec_res = state["execution_result"]

            if exec_res and exec_res.get("success"):
                # Thực thi thành công -> Sang Pedagogical
                break

            # Nếu thực thi thất bại
            if state["attempt_count"] < state["max_attempts"]:
                state = self.corrector_node(state)
                # Tiếp tục vòng lặp để executor chạy lại mã đã sửa
            else:
                # Đã đạt giới hạn số lần sửa
                state["error_log"] = (
                    f"Đã thử tự sửa lỗi {state['max_attempts']} lần nhưng không thành công. "
                    f"Lỗi cuối cùng: {exec_res.get('error')}"
                )
                break

        # 4. Pedagogical Node
        state = self.pedagogical_node(state)

        return state

    def stream_steps(self, problem_text: str) -> Generator[Dict[str, Any], None, AgentState]:
        """
        Bộ chạy theo kiểu Generator phát sinh sự kiện từng bước phục vụ hiển thị trực tiếp trên giao diện Streamlit.
        """
        state: AgentState = {
            "problem": problem_text.strip(),
            "parsed_info": None,
            "sympy_code": None,
            "execution_result": None,
            "correction_history": [],
            "attempt_count": 0,
            "max_attempts": self.max_attempts,
            "final_solution": None,
            "status": "start",
            "error_log": None,
            "agent_logs": []
        }

        # Bước 1
        yield {"step": "parser_start", "message": "🔍 Parser Agent đang phân tích đề bài toán..."}
        state = self.parser_node(state)
        yield {"step": "parser_done", "state": state, "data": state["parsed_info"]}

        # Bước 2
        yield {"step": "symbolic_start", "message": "⚙️ Symbolic Agent đang xây dựng phương trình và sinh mã SymPy..."}
        state = self.symbolic_node(state)
        yield {"step": "symbolic_done", "state": state, "data": state["sympy_code"]}

        # Vòng lặp Sandbox & Correction
        while True:
            yield {
                "step": "executor_start",
                "message": f"🛡️ Safe Sandbox đang thực thi mã (Lần {state['attempt_count'] + 1})..."
            }
            state = self.executor_node(state)
            exec_res = state["execution_result"]
            yield {"step": "executor_done", "state": state, "data": exec_res}

            if exec_res and exec_res.get("success"):
                break

            if state["attempt_count"] < state["max_attempts"]:
                yield {
                    "step": "corrector_start",
                    "message": f"🔄 Phát hiện lỗi! Self-Correction Agent đang phân tích và sửa mã (Lần {state['attempt_count'] + 1}/{state['max_attempts']})..."
                }
                state = self.corrector_node(state)
                yield {"step": "corrector_done", "state": state, "data": state["sympy_code"]}
            else:
                state["error_log"] = (
                    f"Đã thử tự sửa {state['max_attempts']} lần. Lỗi: {exec_res.get('error')}"
                )
                yield {"step": "corrector_failed", "state": state, "data": state["error_log"]}
                break

        # Bước cuối
        yield {"step": "pedagogical_start", "message": "🎓 Pedagogical Agent đang soạn thảo lời giải sư phạm chuẩn SGK THCS..."}
        state = self.pedagogical_node(state)
        yield {"step": "pedagogical_done", "state": state, "data": state["final_solution"]}

        return state


def build_langgraph_workflow(llm_client: BaseLLMClient, timeout: float = 3.0, max_attempts: int = 3):
    """
    Xây dựng StateGraph của LangGraph (nếu thư viện langgraph có sẵn trong môi trường).
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        # Nếu chưa cài langgraph, trả về None để caller dùng MultiAgentMathSolver thuần
        return None

    solver = MultiAgentMathSolver(llm_client, timeout=timeout, max_attempts=max_attempts)
    workflow = StateGraph(AgentState)

    workflow.add_node("parser", solver.parser_node)
    workflow.add_node("symbolic", solver.symbolic_node)
    workflow.add_node("executor", solver.executor_node)
    workflow.add_node("corrector", solver.corrector_node)
    workflow.add_node("pedagogical", solver.pedagogical_node)

    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "symbolic")
    workflow.add_edge("symbolic", "executor")

    def route_after_executor(state: AgentState) -> str:
        res = state.get("execution_result", {})
        if res and res.get("success"):
            return "pedagogical"
        if state.get("attempt_count", 0) < state.get("max_attempts", 3):
            return "corrector"
        if not state.get("error_log"):
            state["error_log"] = (
                f"Đã thử tự sửa lỗi {state.get('max_attempts', 3)} lần nhưng không thành công. "
                f"Lỗi cuối cùng: {res.get('error')}"
            )
        return "pedagogical"

    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "pedagogical": "pedagogical",
            "corrector": "corrector"
        }
    )

    workflow.add_edge("corrector", "executor")
    workflow.add_edge("pedagogical", END)

    return workflow.compile()
