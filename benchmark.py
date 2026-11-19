# -*- coding: utf-8 -*-
"""
Script Đánh giá Hiệu năng Định lượng (Benchmark Runner).
Đo lường các chỉ số cốt lõi của Hệ thống Multi-Agent Giải toán THCS:
1. Accuracy (Độ chính xác so với ground truth).
2. Execution Success Rate - ESR (Tỷ lệ thực thi thành công của mã ký hiệu).
3. Self-Correction Recovery Rate - SCRR (Tỷ lệ tự sửa lỗi phục hồi thành công).
4. Average Latency (Thời gian xử lý trung bình từng câu hỏi).

Hỗ trợ 2 chế độ dữ liệu:
- 'local': Tải bài toán từ file JSON nội bộ (mặc định: data/thcs_problems.json hoặc data/gsm8k_vi_sample.json).
- 'vi_gsm8k': Tải trực tiếp 1.319 câu hỏi từ bộ dữ liệu chuẩn Hugging Face 'hllj/vi_gsm8k' vào RAM.
"""

import argparse
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import config
from src.llm.factory import LLMFactory
from src.workflow import MultiAgentMathSolver


def extract_gsm8k_answer(answer_text: Any) -> Any:
    """
    Trích xuất giá trị đáp số số học từ chuỗi answer của bộ dữ liệu GSM8K / vi_gsm8k.
    Trong định dạng chuẩn GSM8K, đáp số cuối cùng luôn nằm sau ký hiệu '####'.
    Ví dụ: '... Do đó Nam có tất cả 105 viên bi.\n#### 105' -> 105
           '... Tổng chi phí là $1,200.\n#### $1,200' -> 1200
           '#### 3/4' -> '3/4'
    """
    if answer_text is None:
        return None

    text = str(answer_text).strip()
    if "####" in text:
        target = text.split("####")[-1].strip()
    else:
        target = text

    # Loại bỏ dấu phẩy phân tách hàng nghìn (ví dụ '1,200' -> '1200') và khoảng trắng
    cleaned = target.replace(",", "").strip()

    # 1. Thử ép kiểu số nguyên / số thực trực tiếp
    try:
        val_float = float(cleaned)
        if val_float.is_integer():
            return int(val_float)
        return val_float
    except (ValueError, OverflowError):
        pass

    # 2. Xử lý phân số dạng chuỗi "p/q" (ví dụ "3/4", "-1/2")
    if "/" in cleaned:
        frac_parts = cleaned.split("/")
        if len(frac_parts) == 2:
            p_str, q_str = frac_parts[0].strip(), frac_parts[1].strip()
            if p_str.lstrip("+-").isdigit() and q_str.lstrip("+-").isdigit():
                return f"{p_str}/{q_str}"

    # 3. Xử lý trường hợp có ký hiệu tiền tệ hoặc đơn vị đi kèm (ví dụ: "$1200", "45 viên", "25%")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if match:
        try:
            vf = float(match.group(0))
            if vf.is_integer():
                return int(vf)
            return vf
        except (ValueError, OverflowError):
            pass

    return cleaned


def _to_float_val(val: Any) -> Optional[float]:
    """
    Chuyển đổi linh hoạt giá trị thành số thực float phục vụ so khớp số học chính xác.
    Hỗ trợ int, float, chuỗi số, phân số chuỗi ("3/4"), và chuỗi chứa tiền tệ/đơn vị.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    text = str(val).replace(",", "").strip()

    # Thử float trực tiếp
    try:
        return float(text)
    except (ValueError, TypeError):
        pass

    # Thử phân số (ví dụ: "3/4", "-1/2")
    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) == 2 and parts[0].strip().lstrip("+-").isdigit() and parts[1].strip().lstrip("+-").isdigit():
                return float(Fraction(int(parts[0]), int(parts[1])))
    except (ValueError, ZeroDivisionError, TypeError):
        pass

    # Thử bóc tách số từ chuỗi có ký hiệu tiền tệ hoặc đơn vị (ví dụ "$1200", "25%")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except (ValueError, TypeError):
            pass

    return None


def is_answer_close(pred: Any, expected: Any, tol: float = 1e-4) -> bool:
    """
    Kiểm tra xem đáp án dự đoán có khớp với đáp án kỳ vọng hay không.
    Hỗ trợ so sánh số học với sai số dung sai (tolerance), phân số chuỗi,
    tập nghiệm list, và từ điển nghiệm dict.
    """
    if pred is None or expected is None:
        return False

    # Nếu pred là list chứa nhiều nghiệm (FiniteSet / Solve output)
    if isinstance(pred, list):
        for item in pred:
            if is_answer_close(item, expected, tol):
                return True
        return False

    # Nếu pred là dict (chứa các biến và giá trị nghiệm)
    if isinstance(pred, dict):
        for val in pred.values():
            if is_answer_close(val, expected, tol):
                return True
        return False

    # So sánh giá trị số học (hỗ trợ cả phân số và tiền tệ/đơn vị)
    f_pred = _to_float_val(pred)
    f_exp = _to_float_val(expected)
    if f_pred is not None and f_exp is not None:
        return abs(f_pred - f_exp) <= tol or (abs(f_exp) > 0 and abs(f_pred - f_exp) / abs(f_exp) <= tol)

    # So sánh chuỗi nếu không chuyển được sang số
    return str(pred).strip().lower() == str(expected).strip().lower()


def load_benchmark_data(
    dataset_mode: str = "local",
    dataset_path: Optional[Union[str, Path]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Tải danh sách bài toán benchmark từ file nội bộ (local) hoặc tải trực tiếp vào RAM
    từ bộ dữ liệu chuẩn Hugging Face ('vi_gsm8k').
    
    Trả về:
        Tuple[List[Dict[str, Any]], str]: (Danh sách bài toán đã chuẩn hóa, Mô tả tập dữ liệu)
    """
    mode = str(dataset_mode).strip().lower()

    # 1. Chế độ Hugging Face Dataset: hllj/vi_gsm8k
    if mode in ("vi_gsm8k", "hllj/vi_gsm8k"):
        try:
            from datasets import load_dataset
        except (ImportError, ModuleNotFoundError):
            print("\n" + "=" * 80)
            print("[!] LỖI THIẾU THƯ VIỆN: Chưa cài đặt thư viện 'datasets' để tải 'hllj/vi_gsm8k'.")
            print("[*] Hướng dẫn cài đặt:")
            print("    - Trên máy tính cục bộ: pip install datasets")
            print("    - Trên Google Colab:    !pip install -q datasets")
            print("=" * 80 + "\n")
            sys.exit(1)

        print("[*] Đang tải trực tiếp bộ dữ liệu 'hllj/vi_gsm8k' (split='test') từ Hugging Face vào RAM...")
        try:
            ds = load_dataset("hllj/vi_gsm8k", split="test")
        except Exception as err:
            print(f"[!] Lỗi khi kết nối hoặc tải dữ liệu từ Hugging Face Hub: {err}")
            sys.exit(1)

        problems: List[Dict[str, Any]] = []
        for idx, row in enumerate(ds, 1):
            q = row.get("question", "")
            raw_ans = row.get("answer", "")
            extracted_ans = extract_gsm8k_answer(raw_ans)
            problems.append({
                "id": f"vi_gsm8k_{idx:04d}",
                "category": "GSM8K Tiếng Việt",
                "question": q,
                "expected_answer": extracted_ans,
                "raw_answer": raw_ans
            })

        dataset_desc = f"Hugging Face: hllj/vi_gsm8k (split='test', tổng số {len(problems)} câu hỏi)"
        return problems, dataset_desc

    # 2. Chế độ file cục bộ (local JSON)
    target_file: Optional[Path] = None
    if mode in ("local", "thcs", "thcs_problems"):
        if dataset_path:
            target_file = Path(dataset_path)
        else:
            target_file = config.DEFAULT_BENCHMARK_DATASET
    elif mode in ("gsm8k", "gsm8k_sample", "gsm8k_vi_sample", "gsm8k_local"):
        if dataset_path:
            target_file = Path(dataset_path)
        else:
            target_file = getattr(config, "GSM8K_VI_DATASET", config.DATA_DIR / "gsm8k_vi_sample.json")
    else:
        # Nếu người dùng truyền thẳng đường dẫn file JSON vào --dataset
        target_file = Path(dataset_mode)

    if not target_file.exists():
        print(f"[!] Lỗi: Không tìm thấy tệp dữ liệu tại {target_file}")
        print("    👉 Gợi ý: Dùng --dataset local (mặc định), --dataset gsm8k hoặc --dataset vi_gsm8k (Hugging Face).")
        sys.exit(1)

    with open(target_file, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # Chuẩn hóa nếu tập JSON bên ngoài chỉ có trường "answer" thay vì "expected_answer"
    for p in problems:
        if "expected_answer" not in p and "answer" in p:
            p["expected_answer"] = extract_gsm8k_answer(p["answer"])

    dataset_desc = f"Tệp cục bộ: {target_file} ({len(problems)} câu hỏi)"
    return problems, dataset_desc


def run_benchmark(
    dataset: str = "local",
    dataset_path: Optional[Union[str, Path]] = None,
    provider: str = "gemini",
    model: str = None,
    api_key: str = None,
    limit: int = None,
    timeout: float = 3.0,
    max_attempts: int = 3,
    output_file: str = "benchmark_results.json",
    delay: float = 0.5
) -> Dict[str, Any]:
    """
    Chạy đánh giá định lượng trên tập dữ liệu được chỉ định (local hoặc vi_gsm8k).
    """
    # Tải và chuẩn bị bộ dữ liệu
    problems, dataset_desc = load_benchmark_data(dataset_mode=dataset, dataset_path=dataset_path)

    if limit and limit > 0:
        problems = problems[:limit]

    total_problems = len(problems)

    # Chuẩn hóa Provider và Model mặc định
    selected_provider = (provider or config.DEFAULT_PROVIDER).lower()
    selected_model = model
    selected_api_key = api_key

    if selected_provider == "gemini":
        selected_model = selected_model or config.DEFAULT_GEMINI_MODEL
        selected_api_key = selected_api_key or config.GEMINI_API_KEY
    elif selected_provider == "openrouter":
        selected_model = selected_model or config.DEFAULT_OPENROUTER_MODEL
        selected_api_key = selected_api_key or config.OPENROUTER_API_KEY

    print("=" * 80)
    print("      HỆ THỐNG ĐÁNH GIÁ ĐỊNH LƯỢNG MULTI-AGENT GIẢI TOÁN THCS (BENCHMARK)     ")
    print("=" * 80)
    print(f"[*] Tập dữ liệu : {dataset_desc}")
    print(f"[*] LLM Provider: {selected_provider} | Model: {selected_model}")
    print(f"[*] Sandbox Timeout: {timeout}s | Max Correction Attempts: {max_attempts}")
    print(f"[*] Tổng số câu hỏi kiểm thử: {total_problems}")
    print("-" * 80)

    # Khởi tạo Solver
    llm_client = LLMFactory.create_client(
        provider=selected_provider,
        api_key=selected_api_key,
        model=selected_model
    )
    solver = MultiAgentMathSolver(
        llm_client=llm_client,
        timeout=timeout,
        max_attempts=max_attempts
    )

    results_detail: List[Dict[str, Any]] = []
    correct_count = 0
    exec_success_count = 0
    first_try_failed_count = 0
    recovered_count = 0
    total_time = 0.0

    for idx, p in enumerate(problems, 1):
        pid = p.get("id", f"p_{idx}")
        question = p.get("question", "")
        expected = p.get("expected_answer")
        category = p.get("category", "Chung")

        print(f"\n[{idx}/{total_problems}] Đang giải [{pid}] ({category})...")
        t_start = time.time()

        try:
            state = solver.run(question)
            duration = time.time() - t_start
            total_time += duration

            exec_res = state.get("execution_result") or {}
            exec_ok = bool(exec_res.get("success", False))
            attempts = state.get("attempt_count", 0)
            pred_output = exec_res.get("output")

            is_correct = is_answer_close(pred_output, expected)

            # Thống kê ESR
            if exec_ok:
                exec_success_count += 1

            # Thống kê SCRR
            if attempts > 0:
                first_try_failed_count += 1
                if exec_ok and is_correct:
                    recovered_count += 1

            if is_correct:
                correct_count += 1

            status_symbol = "✓ ĐÚNG" if is_correct else "✗ SAI"
            print(f"    -> Dự đoán: {pred_output} | Kỳ vọng: {expected} | Kết quả: {status_symbol} ({duration:.2f}s)")
            if attempts > 0:
                print(f"    -> [Tự sửa lỗi] Đã qua {attempts} lần sửa mã.")

            results_detail.append({
                "id": pid,
                "category": category,
                "question": question,
                "expected": expected,
                "predicted": pred_output,
                "is_correct": is_correct,
                "exec_success": exec_ok,
                "attempts": attempts,
                "duration": duration,
                "error": exec_res.get("error")
            })

        except Exception as ex:
            duration = time.time() - t_start
            total_time += duration
            print(f"    -> [!] Lỗi không xử lý được: {ex}")
            results_detail.append({
                "id": pid,
                "category": category,
                "question": question,
                "expected": expected,
                "predicted": None,
                "is_correct": False,
                "exec_success": False,
                "attempts": 0,
                "duration": duration,
                "error": str(ex)
            })

        if delay > 0 and idx < total_problems:
            time.sleep(delay)

    # Tính toán các chỉ số định lượng
    accuracy = (correct_count / total_problems) * 100 if total_problems > 0 else 0.0
    esr = (exec_success_count / total_problems) * 100 if total_problems > 0 else 0.0
    scrr = (recovered_count / first_try_failed_count) * 100 if first_try_failed_count > 0 else 100.0
    avg_latency = total_time / total_problems if total_problems > 0 else 0.0

    report = {
        "dataset": dataset_desc,
        "dataset_mode": dataset,
        "total_problems": total_problems,
        "metrics": {
            "accuracy_percent": round(accuracy, 2),
            "execution_success_rate_percent": round(esr, 2),
            "self_correction_recovery_rate_percent": round(scrr, 2),
            "first_try_failed_cases": first_try_failed_count,
            "successfully_recovered_cases": recovered_count,
            "average_latency_seconds": round(avg_latency, 2),
            "total_benchmark_time_seconds": round(total_time, 2)
        },
        "config": {
            "provider": provider,
            "model": model,
            "sandbox_timeout": timeout,
            "max_attempts": max_attempts
        },
        "details": results_detail
    }

    # In Bảng Báo cáo Tổng kết
    print("\n" + "=" * 80)
    print("                     BÁO CÁO KẾT QUẢ ĐỊNH LƯỢNG CUỐI CÙNG                     ")
    print("=" * 80)
    print(f" 1. Tập dữ liệu kiểm nghiệm       : {dataset_desc}")
    print(f" 2. Tổng số câu hỏi kiểm nghiệm   : {total_problems}")
    print(f" 3. Độ chính xác (Accuracy)       : {accuracy:.2f}% ({correct_count}/{total_problems})")
    print(f" 4. Tỷ lệ thực thi mã (ESR)       : {esr:.2f}% ({exec_success_count}/{total_problems})")
    print(f" 5. Tỷ lệ phục hồi lỗi (SCRR)    : {scrr:.2f}% ({recovered_count}/{first_try_failed_count} ca gặp sự cố)")
    print(f" 6. Thời gian phản hồi TB (Latency): {avg_latency:.2f}s / bài toán")
    print("=" * 80)

    # Lưu kết quả ra file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã xuất kết quả chi tiết ra tệp: {out_path.resolve()}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Benchmark Định lượng Multi-Agent Toán THCS")
    parser.add_argument("--provider", type=str, default=config.DEFAULT_PROVIDER, help="Nhà cung cấp: gemini, openrouter, mock")
    parser.add_argument("--model", type=str, default=None, help="Tên model LLM (mặc định theo provider)")
    parser.add_argument("--api-key", type=str, default=None, help="Khóa API (Gemini hoặc OpenRouter)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="local",
        help="Lựa chọn tập dữ liệu: 'local' (data/thcs_problems.json), 'gsm8k' (data/gsm8k_vi_sample.json), 'vi_gsm8k' (Hugging Face hllj/vi_gsm8k) hoặc đường dẫn tệp JSON"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Đường dẫn tệp JSON tùy chọn khi dùng chế độ local (mặc định: data/thcs_problems.json)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng câu hỏi cần chạy")
    parser.add_argument("--timeout", type=float, default=config.SANDBOX_TIMEOUT_SECONDS, help="Timeout cho Sandbox (giây)")
    parser.add_argument("--max-attempts", type=int, default=config.MAX_CORRECTION_ATTEMPTS, help="Số lần tự sửa tối đa")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Tên tệp JSON xuất báo cáo")
    parser.add_argument("--delay", type=float, default=0.5, help="Độ trễ giữa các câu hỏi để tránh Rate Limit (giây)")

    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        output_file=args.output,
        delay=args.delay
    )
