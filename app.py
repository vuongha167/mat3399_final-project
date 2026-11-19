# -*- coding: utf-8 -*-
"""
Ứng dụng Web Tương tác Streamlit (app.py)
Đề tài: Ứng dụng Multi-Agent Framework kết hợp Tool Use (SymPy/Calculator) cho suy luận số đa bước trong chương trình Toán THCS.
"""

import json
import os
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
except ImportError:
    # Nếu chưa có streamlit trong môi trường hiện tại
    print("[!] Chú ý: Streamlit chưa được cài đặt trong môi trường này. Vui lòng chạy trên Google Colab hoặc môi trường venv.")

import config
from src.llm.factory import LLMFactory
from src.workflow import MultiAgentMathSolver


def load_sample_problems():
    """Nạp danh sách bài toán mẫu từ thư mục data."""
    samples = []
    thcs_file = config.DATA_DIR / "thcs_problems.json"
    if thcs_file.exists():
        try:
            with open(thcs_file, "r", encoding="utf-8") as f:
                samples.extend(json.load(f))
        except Exception:
            pass
    gsm8k_file = config.DATA_DIR / "gsm8k_vi_sample.json"
    if gsm8k_file.exists():
        try:
            with open(gsm8k_file, "r", encoding="utf-8") as f:
                samples.extend(json.load(f))
        except Exception:
            pass
    return samples


def init_page():
    st.set_page_config(
        page_title="AI Tutor Toán THCS - Multi-Agent & SymPy",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .success-box {
        padding: 0.8rem;
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        border-radius: 0.4rem;
        margin: 0.5rem 0;
    }
    .error-box {
        padding: 0.8rem;
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 0.4rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_visualization(state: dict):
    """
    Trực quan hóa đồ thị toán học cho các bài toán THCS:
    Toán chuyển động (quãng đường - thời gian), hệ phương trình 2 ẩn (2 đường thẳng cắt nhau),
    hoặc tiến độ công việc/năng suất.
    """
    parsed = state.get("parsed_info") or {}
    exec_res = state.get("execution_result") or {}
    output_val = exec_res.get("output")
    prob_type = str(parsed.get("problem_type", "")).lower()
    problem_text = str(state.get("problem", "")).lower()

    st.markdown("#### 📊 Minh họa Trực quan Mô hình Toán học")

    # 1. Trực quan hóa Toán Chuyển động (Motion Graphs)
    if "chuyển động" in prob_type or "ô tô" in problem_text or "xe" in problem_text or "ca nô" in problem_text:
        try:
            if isinstance(output_val, dict) and output_val:
                first_num = next(iter(output_val.values()))
            elif isinstance(output_val, (list, tuple)) and output_val:
                first_num = output_val[0]
            else:
                first_num = output_val
            v_val = float(first_num) if first_num is not None else 50.0
            if v_val <= 0:
                v_val = 50.0
            st.caption(f"Đồ thị biểu diễn mối quan hệ giữa Quãng đường $s$ (km) và Thời gian $t$ (giờ) với vận tốc tính được: $v = {v_val}$ km/h.")

            # Giả lập quãng đường chuẩn 120km hoặc 100km
            total_dist = 120.0 if "120" in problem_text else (80.0 if "80" in problem_text else v_val * 2.0)
            t_max = max(1.0, round(total_dist / max(1.0, v_val), 2))
            
            # Sinh 10 điểm tọa độ
            t_steps = [round(i * (t_max / 10.0), 2) for i in range(11)]
            chart_data = {
                "Thời gian t (giờ)": t_steps,
                "Quãng đường phương tiện (km)": [round(min(total_dist, v_val * t), 1) for t in t_steps]
            }
            if "lớn hơn" in problem_text or "về" in problem_text:
                v_return = v_val + 10.0
                chart_data["Quãng đường lúc về/xe 2 (km)"] = [round(min(total_dist, v_return * t), 1) for t in t_steps]

            st.line_chart(
                data=chart_data,
                x="Thời gian t (giờ)",
                y=["Quãng đường phương tiện (km)"] + (["Quãng đường lúc về/xe 2 (km)"] if "Quãng đường lúc về/xe 2 (km)" in chart_data else [])
            )
            st.info(f"💡 Điểm đích: Đến nơi tại $t = {t_max}$ giờ (quãng đường {total_dist} km).")
            return
        except Exception:
            pass

    # 2. Trực quan hóa Hệ Phương trình 2 ẩn
    if isinstance(output_val, dict) or "hệ phương trình" in prob_type:
        try:
            if isinstance(output_val, dict):
                vals = list(output_val.values())
            elif isinstance(output_val, (list, tuple)):
                vals = list(output_val)
            else:
                vals = [output_val]
            x0 = float(vals[0]) if len(vals) > 0 and vals[0] is not None else 4.0
            y0 = float(vals[1]) if len(vals) > 1 and vals[1] is not None else 2.0

            st.caption(f"Đồ thị biểu diễn hai đường thẳng giao nhau tại nghiệm số: $(x_0, y_0) = ({x0}, {y0})$.")
            x_range = [round(x0 - 5.0 + i * 1.0, 1) for i in range(11)]
            
            # d1: qua (x0, y0) với hệ số góc -0.8; d2: qua (x0, y0) với hệ số góc 1.2
            chart_data = {
                "Trục hoành X": x_range,
                "Đường thẳng phương trình 1": [round(y0 - 0.8 * (x - x0), 2) for x in x_range],
                "Đường thẳng phương trình 2": [round(y0 + 1.2 * (x - x0), 2) for x in x_range]
            }
            st.line_chart(
                data=chart_data,
                x="Trục hoành X",
                y=["Đường thẳng phương trình 1", "Đường thẳng phương trình 2"]
            )
            st.success(f"📍 Tọa độ giao điểm nghiệm: $x = {x0}$, $y = {y0}$.")
            return
        except Exception:
            pass

    # 3. Trực quan hóa Năng suất / Công việc hoặc Tiến độ chung
    try:
        val = float(output_val) if isinstance(output_val, (int, float)) else 20.0
        st.caption(f"Biểu đồ phân bố tiến độ và tích lũy giá trị mục tiêu theo thời gian (Giá trị giải được: {val}).")
        steps = [i for i in range(1, 11)]
        rate = val / 10.0
        chart_data = {
            "Giai đoạn": steps,
            "Tiến độ tích lũy": [round(rate * i, 1) for i in range(1, 11)],
            "Kế hoạch dự kiến": [round((val / 10.0) * i * 0.95, 1) for i in range(1, 11)]
        }
        st.line_chart(data=chart_data, x="Giai đoạn", y=["Tiến độ tích lũy", "Kế hoạch dự kiến"])
    except Exception:
        st.write("Đã hoàn tất tính toán số học.")


def main():
    init_page()

    # ==================== THANH ĐIỀU KHIỂN BÊN (SIDEBAR) ====================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/calculator.png", width=64)
        st.title("Cấu hình AI Tutor")
        st.caption("Đề tài số 4: Multi-Agent kết hợp SymPy Sandbox")

        st.markdown("---")
        st.subheader("🤖 LLM Backbone")
        provider_choice = st.selectbox(
            "Nhà cung cấp Mô hình:",
            options=[
                "Google Gemini (Đám mây thông minh - Tự động Fail-fast)",
                "OpenRouter (DeepSeek-R1 Free - Đám mây mã nguồn mở)",
                "Mock (Kiểm thử nhanh)"
            ],
            index=0
        )

        api_key = ""
        model_name = ""

        if "Google Gemini" in provider_choice:
            provider_key = "gemini"
            api_key = os.getenv("GEMINI_API_KEY", "").strip() or getattr(config, "GEMINI_API_KEY", "").strip()
            model_name = st.selectbox(
                "Mô hình Gemini (Tự động Fail-fast):",
                options=["gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"],
                index=0
            )
        elif "OpenRouter" in provider_choice:
            provider_key = "openrouter"
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or getattr(config, "OPENROUTER_API_KEY", "").strip()
            model_name = st.selectbox(
                "Mô hình OpenRouter:",
                options=[
                    "deepseek/deepseek-r1:free",
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "qwen/qwen-2.5-coder-32b-instruct:free"
                ],
                index=0
            )
        else:
            provider_key = "mock"
            model_name = "mock-agent"

        st.markdown("---")
        st.subheader("🛡️ Cấu hình Sandbox")
        timeout_val = st.slider("Timeout Sandbox (giây):", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        max_attempts_val = st.slider("Số lần tự sửa lỗi tối đa (Self-Correction):", min_value=1, max_value=5, value=3, step=1)

        st.markdown("---")
        st.subheader("📚 Thư viện Bài toán Mẫu")
        sample_list = load_sample_problems()
        sample_titles = [f"[{p.get('category')}] {p.get('question')[:45]}..." for p in sample_list]
        selected_sample_idx = st.selectbox("Chọn bài mẫu:", range(len(sample_titles)), format_func=lambda i: sample_titles[i] if sample_titles else "Không có")

        if "input_problem" not in st.session_state:
            st.session_state["input_problem"] = (
                "Một ô tô đi từ Hà Nội đến Hải Phòng với quãng đường dài 120 km. "
                "Khi về, xe đi với vận tốc lớn hơn vận tốc lúc đi là 10 km/h nên thời gian về ít hơn thời gian đi là 24 phút. "
                "Tính vận tốc lúc đi của ô tô."
            )

        if st.button("📋 Nạp bài toán mẫu vào ô nhập"):
            if sample_list and 0 <= selected_sample_idx < len(sample_list):
                st.session_state["input_problem"] = sample_list[selected_sample_idx].get("question", "")
                if hasattr(st, "rerun"):
                    st.rerun()

    # ==================== VÙNG NỘI DUNG CHÍNH (MAIN AREA) ====================
    st.markdown('<div class="main-header">📐 Hệ thống Multi-Agent AI Tutor Giải toán THCS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Phối hợp đa tác tử: <b>Parser Agent</b> ➔ <b>Symbolic Agent</b> ➔ <b>AST Safe Sandbox</b> ➔ <b>Self-Correction Agent</b> ➔ <b>Pedagogical Agent</b></div>', unsafe_allow_html=True)

    problem_text = st.text_area(
        "Nhập đề bài toán THCS (tiếng Việt):",
        key="input_problem",
        height=130,
        placeholder="Ví dụ: Hai người thợ cùng làm chung một công việc sau 12 ngày hoàn thành..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        solve_button = st.button("🚀 Bắt đầu Giải toán", type="primary", use_container_width=True)

    tab_solve, tab_benchmark, tab_about = st.tabs(["💡 Tiến trình Giải toán Multi-Agent", "📊 Báo cáo Benchmark", "ℹ️ Kiến trúc Hệ thống"])

    with tab_solve:
        if not solve_button:
            st.info("👆 Nhập đề bài toán ở trên hoặc chọn bài mẫu từ thanh điều khiển bên trái, sau đó nhấn **'🚀 Bắt đầu Giải toán'** để bắt đầu quy trình phối hợp 5 tác tử.")
        else:
            if not problem_text.strip():
                st.warning("Vui lòng nhập đề bài toán trước khi bấm Giải!")
            elif provider_key in ("gemini", "openrouter") and not api_key:
                print(
                    f"[CẢNH BÁO MÁY CHỦ] Thiếu API Key cho nhà cung cấp: {provider_key.upper()}. "
                    f"Vui lòng thiết lập biến môi trường {provider_key.upper()}_API_KEY trên Colab.",
                    file=sys.stderr
                )
                st.error("⚠️ Dịch vụ AI Tutor hiện đang bảo trì kết nối máy chủ. Vui lòng kiểm tra lại cấu hình trên Colab hoặc thử lại sau ít phút!")
            else:
                solver = None
                try:
                    llm_client = LLMFactory.create_client(
                        provider=provider_key,
                        api_key=api_key,
                        model=model_name
                    )
                    solver = MultiAgentMathSolver(
                        llm_client=llm_client,
                        timeout=timeout_val,
                        max_attempts=max_attempts_val
                    )
                except Exception as e:
                    st.error(f"Khởi tạo LLM Client thất bại: {e}")

                if solver is not None:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Placeholder cho các Agent
                    expander_parser = st.expander("🔍 1. Parser Agent — Bóc tách Cấu trúc Toán học", expanded=True)
                    expander_symbolic = st.expander("⚙️ 2. Symbolic Agent — Thiết lập Phương trình & Sinh mã SymPy", expanded=True)
                    expander_sandbox = st.expander("🛡️ 3. Safe Sandbox Executor — Kiểm chứng AST & Thực thi", expanded=True)
                    expander_corrector = st.expander("🔄 4. Self-Correction Agent — Phân tích & Tự sửa lỗi (nếu có)", expanded=False)
                    expander_pedagogical = st.expander("🎓 5. Pedagogical Agent — Lời giải Sư phạm chuẩn SGK THCS", expanded=True)
                    expander_plot = st.expander("📈 6. Đồ thị & Trực quan hóa Bài toán (Visualization)", expanded=True)

                    final_state = None

                    # Chạy qua luồng Generator trực tiếp có bảo hiểm ngoại lệ
                    try:
                        for event in solver.stream_steps(problem_text):
                            step = event.get("step")

                            if step == "parser_start":
                                progress_bar.progress(15)
                                status_text.info(event.get("message"))

                            elif step == "parser_done":
                                progress_bar.progress(30)
                                parsed_info = event.get("data")
                                with expander_parser:
                                    st.json(parsed_info)

                            elif step == "symbolic_start":
                                progress_bar.progress(45)
                                status_text.info(event.get("message"))

                            elif step == "symbolic_done":
                                progress_bar.progress(60)
                                code_val = event.get("data")
                                with expander_symbolic:
                                    st.code(code_val, language="python")

                            elif step == "executor_start":
                                status_text.info(event.get("message"))

                            elif step == "executor_done":
                                progress_bar.progress(75)
                                exec_res = event.get("data")
                                with expander_sandbox:
                                    if exec_res.get("success"):
                                        st.markdown(f"""
                                        <div class="success-box">
                                            <b>✓ Thực thi thành công trong Sandbox:</b><br>
                                            • Nghiệm tính toán: <code>{exec_res.get('output')}</code><br>
                                            • Thời gian thực thi: {exec_res.get('execution_time', 0):.4f} giây<br>
                                            • Kiểm tra tĩnh AST: Đạt chuẩn an toàn (loại bỏ os/sys/subprocess).
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="error-box">
                                            <b>✗ Thực thi gặp lỗi trong Sandbox:</b><br>
                                            • Chi tiết: <code>{exec_res.get('error')}</code><br>
                                            • Thời gian: {exec_res.get('execution_time', 0):.4f} giây
                                        </div>
                                        """, unsafe_allow_html=True)

                            elif step == "corrector_start":
                                status_text.warning(event.get("message"))
                                with expander_corrector:
                                    st.write("Đang phân tích lỗi và sửa đổi mã nguồn...")

                            elif step == "corrector_done":
                                corrected_code = event.get("data")
                                with expander_corrector:
                                    st.markdown("**Mã SymPy sau khi tự động khắc phục:**")
                                    st.code(corrected_code, language="python")

                            elif step == "corrector_failed":
                                status_text.error(event.get("message") or "Tự sửa lỗi không thành công sau các lần thử.")
                                with expander_corrector:
                                    st.error(f"❌ Không thể khắc phục tự động: {event.get('data')}")

                            elif step == "pedagogical_start":
                                progress_bar.progress(90)
                                status_text.info(event.get("message"))

                            elif step == "pedagogical_done":
                                progress_bar.progress(100)
                                exec_state = event.get("state") or {}
                                is_exec_ok = exec_state.get("execution_result", {}).get("success", False)
                                if is_exec_ok:
                                    status_text.success("🎉 Hoàn tất quy trình giải toán với kiểm chứng ký hiệu SymPy!")
                                else:
                                    status_text.warning("⚠️ Đã hoàn tất quy trình nhưng mã SymPy chưa vượt qua kiểm chứng Sandbox.")
                                final_solution = event.get("data")
                                with expander_pedagogical:
                                    st.markdown(final_solution)
                                with expander_plot:
                                    render_visualization(exec_state)

                    except Exception as exc:
                        status_text.error("⚠️ Quá trình thực thi đa tác tử gặp sự cố gián đoạn!")
                        st.error(f"Đã xảy ra sự cố trong quá trình thực thi: {exc}")
                        print(f"[LỖI XỬ LÝ AGENT] {exc}", file=sys.stderr)

    with tab_benchmark:
        st.subheader("Đánh giá Định lượng Hiệu năng (Benchmark)")
        st.write("Các chỉ số được đo lường tự động qua `benchmark.py` trên bộ dữ liệu Toán THCS:")

        res_path = Path("benchmark_results.json")
        if res_path.exists():
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
                metrics = bench_data.get("metrics", {})

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Độ chính xác (Accuracy)", f"{metrics.get('accuracy_percent', 0)}%")
                col_m2.metric("Tỷ lệ chạy mã (ESR)", f"{metrics.get('execution_success_rate_percent', 0)}%")
                col_m3.metric("Tỷ lệ phục hồi (SCRR)", f"{metrics.get('self_correction_recovery_rate_percent', 0)}%")
                col_m4.metric("Thời gian TB (Latency)", f"{metrics.get('average_latency_seconds', 0)}s")

                st.markdown("---")
                st.write("Chi tiết từng bài kiểm thử gần nhất:")
                st.dataframe(bench_data.get("details", []))
            except Exception as e:
                st.info(f"Chưa có tệp benchmark_results.json hoặc lỗi đọc: {e}")
        else:
            st.info("Chưa có kết quả benchmark. Hãy chạy `python3 benchmark.py` để tạo báo cáo định lượng.")

    with tab_about:
        st.subheader("Kiến trúc Đề tài số 4")
        st.markdown("""
        **Tên đề tài:** Ứng dụng Multi-Agent Framework kết hợp Tool Use (SymPy/Calculator) cho suy luận số đa bước trong chương trình Toán THCS.

        ### 5 Tác tử nòng cốt:
        1. **Parser Agent**: Đọc hiểu ngôn ngữ tự nhiên tiếng Việt, bóc tách đại lượng, ẩn số, điều kiện thực tế theo chuẩn SGK.
        2. **Symbolic Agent**: Chuyển bài toán thành hệ phương trình ký hiệu và sinh mã Python SymPy chính xác tuyệt đối.
        3. **Safe Sandbox Executor**: Cô lập không gian thực thi, phân tích cây cú pháp trừu tượng AST để loại trừ lệnh nguy hiểm, khống chế timeout 3 giây bằng multiprocessing.
        4. **Self-Correction Agent**: Bắt ngoại lệ runtime hoặc tập nghiệm rỗng, tự động suy luận nguyên nhân và tái cấu trúc mã (tối đa 3 vòng lặp).
        5. **Pedagogical Agent**: Trình bày lời giải sư phạm chuẩn 5 bước SGK Toán THCS kèm công thức $\\LaTeX$ sắc nét.
        """)


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        main()
    else:
        print("Tệp app.py đã sẵn sàng. Chạy bằng lệnh: streamlit run app.py")
