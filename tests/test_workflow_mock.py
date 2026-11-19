# -*- coding: utf-8 -*-
"""
Kiểm thử tích hợp luồng Multi-Agent Workflow với Mock LLM (test_workflow_mock.py).
Đảm bảo luồng tuần hoàn giữa Parser -> Symbolic -> Sandbox -> Corrector -> Pedagogical
hoạt động chính xác, bắt đúng lỗi và phục hồi thành công.
"""

import unittest
from src.llm.client import MockLLMClient
from src.workflow import MultiAgentMathSolver


class TestMultiAgentWorkflow(unittest.TestCase):

    def test_happy_path_workflow(self):
        """Kiểm thử luồng chuẩn khi mã SymPy chạy thành công ngay lần 1."""
        responses = {
            "Parser Agent": '{"problem_type": "Toán chuyển động", "variables": [{"name": "x", "condition": "x > 0"}]}',
            "Symbolic Agent": """```python
from sympy import symbols, Eq, solve
x = symbols('x')
eq = Eq(2*x + 10, 50)
sol = solve(eq, x)
result = sol[0]
```""",
            "Pedagogical Agent": "### Lời giải chi tiết:\nGọi $x$ là vận tốc..."
        }

        mock_llm = MockLLMClient(predefined_responses=responses)
        solver = MultiAgentMathSolver(llm_client=mock_llm)

        problem = "Tìm một số biết rằng 2 lần số đó cộng 10 thì bằng 50."
        final_state = solver.run(problem)

        self.assertEqual(final_state["status"], "completed")
        self.assertIsNotNone(final_state["execution_result"])
        self.assertTrue(final_state["execution_result"]["success"])
        # Nghiệm là 20
        self.assertEqual(final_state["execution_result"]["output"], 20)
        self.assertEqual(final_state["attempt_count"], 0)
        self.assertIn("Lời giải chi tiết", final_state["final_solution"])

    def test_self_correction_recovery(self):
        """Kiểm thử luồng phục hồi khi lần 1 bị lỗi, Corrector sửa lại mã thành công."""
        # Lần 1: Symbolic Agent sinh mã bị lỗi SyntaxError hoặc DivisionByZero
        # Lần 2: Corrector Agent sinh mã sửa đổi hợp lệ
        mock_llm = MockLLMClient()

        # Giả lập phản hồi tuần tự bằng hàm mock tùy chỉnh
        call_count = {"symbolic": 0, "corrector": 0}

        def mock_generate(prompt, system_instruction=None, temperature=0.2, **kwargs):
            if system_instruction and "Parser Agent" in system_instruction:
                return '{"problem_type": "Toán chia quà", "variables": [{"name": "n"}]}'
            elif system_instruction and "Symbolic Agent" in system_instruction:
                call_count["symbolic"] += 1
                # Sinh mã bị lỗi ZeroDivisionError
                return "```python\nx = 10 / 0\nresult = x\n```"
            elif system_instruction and "Self-Correction Agent" in system_instruction:
                call_count["corrector"] += 1
                # Sửa thành mã chuẩn
                return """```python
from sympy import symbols, Eq, solve
x = symbols('x')
eq = Eq(3*x, 30)
result = solve(eq, x)[0]
```"""
            elif system_instruction and "Pedagogical Agent" in system_instruction:
                return "Lời giải sư phạm hoàn tất với nghiệm $x = 10$."
            return "Phản hồi mặc định"

        mock_llm.generate = mock_generate

        solver = MultiAgentMathSolver(llm_client=mock_llm, max_attempts=3)
        final_state = solver.run("Một bài toán mẫu cần tự sửa lỗi.")

        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["attempt_count"], 1)
        self.assertEqual(len(final_state["correction_history"]), 1)
        self.assertTrue(final_state["execution_result"]["success"])
        self.assertEqual(final_state["execution_result"]["output"], 10)
        self.assertIn("nghiệm $x = 10$", final_state["final_solution"])

    def test_generator_stream_steps(self):
        """Kiểm thử luồng generator stream_steps phục vụ Streamlit."""
        responses = {
            "Parser Agent": '{"problem_type": "Toán tỉ số"}',
            "Symbolic Agent": "```python\nresult = 42\n```",
            "Pedagogical Agent": "Đáp số là 42."
        }
        mock_llm = MockLLMClient(predefined_responses=responses)
        solver = MultiAgentMathSolver(llm_client=mock_llm)

        steps_emitted = []
        for event in solver.stream_steps("Bài toán kiểm tra generator"):
            steps_emitted.append(event["step"])

        expected_steps = [
            "parser_start", "parser_done",
            "symbolic_start", "symbolic_done",
            "executor_start", "executor_done",
            "pedagogical_start", "pedagogical_done"
        ]
        for exp in expected_steps:
            self.assertIn(exp, steps_emitted)

    def test_linear_system_2d_workflow(self):
        """Kiểm thử luồng giải bài toán hệ phương trình 2 ẩn qua workflow."""
        responses = {
            "Parser Agent": '{"problem_type": "Hệ phương trình bậc nhất hai ẩn", "variables": [{"name": "x"}, {"name": "y"}]}',
            "Symbolic Agent": """```python
from sympy import symbols, Eq, solve
x, y = symbols('x y')
eq1 = Eq(x + y, 100)
eq2 = Eq(x - y, 28)
result = solve([eq1, eq2], [x, y])
```""",
            "Pedagogical Agent": "Tìm được nghiệm $(x, y) = (64, 36)$."
        }
        mock_llm = MockLLMClient(predefined_responses=responses)
        solver = MultiAgentMathSolver(llm_client=mock_llm)

        problem = "Tìm hai số biết tổng bằng 100 và hiệu bằng 28."
        final_state = solver.run(problem, prefer_langgraph=True)

        self.assertEqual(final_state["status"], "completed")
        self.assertTrue(final_state["execution_result"]["success"])
        res_dict = final_state["execution_result"]["output"]
        self.assertIsInstance(res_dict, dict)
        self.assertEqual(res_dict.get("x"), 64)
        self.assertEqual(res_dict.get("y"), 36)


if __name__ == "__main__":
    unittest.main()
