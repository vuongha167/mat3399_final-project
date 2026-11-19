# KIẾN TRÚC ĐA TÁC TỬ (MULTI-AGENT ARCHITECTURE) - ĐỀ TÀI SỐ 4

**Đề tài:** Ứng dụng Multi-Agent Framework kết hợp Tool Use (SymPy/Calculator) cho suy luận số đa bước trong chương trình Toán THCS  
**Mục tiêu nghiên cứu:** Xây dựng hệ thống tác tử giải toán có kiểm chứng ký hiệu (**Symbolic Verification**), khắc phục triệt để hiện tượng ảo giác (hallucination) trong suy luận số học đa bước của mô hình ngôn ngữ lớn (LLM).

---

## 1. Tổng quan Kiến trúc Hệ thống

Hệ thống hoạt động theo mô hình đồ thị trạng thái (**State Graph**) có chu trình tuần hoàn điều phối qua `LangGraph` (kèm pure Python runner dự phòng), kết hợp 5 thành phần chuyên biệt:

```
                  ┌──────────────────────┐
                  │    Đề bài Toán THCS   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    1. PARSER AGENT   │
                  │ (Trích xuất cấu trúc)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  2. SYMBOLIC AGENT   │
                  │  (Sinh mã SymPy)     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ 3. SAFE AST SANDBOX  │
                  │  (Thực thi & Timeout)│
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
       [Lỗi / Ngoại lệ]             [Thành công]
              │                             │
              ▼                             │
    ┌───────────────────┐                   │
    │ 4. SELF-CORRECTION│                   │
    │  (Sửa mã tối đa   │                   │
    │     3 lần lặp)    │                   │
    └─────────┬─────────┘                   │
              │ (Quay lại Sandbox)          │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ 5. PEDAGOGICAL AGENT │
                  │(Diễn giải 5 bước SGK)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Lời giải LaTeX &    │
                  │  Đáp số đã xác thực  │
                  └──────────────────────┘
```

---

## 2. Chi tiết Các Tác tử (Agents Specification)

### 2.1. Parser Agent (Tác tử Phân tích Cấu trúc)
- **Nhiệm vụ:** Đọc hiểu văn bản tự nhiên đề bài toán THCS bằng tiếng Việt (các bài toán lập phương trình/hệ phương trình: chuyển động, làm chung làm riêng, năng suất, nồng độ dung dịch, kinh tế thực tế, hình học).
- **Đầu ra:** Chuẩn hóa sang cấu trúc JSON chứa:
  - `problem_type`: Phân loại bài toán theo chương trình giáo dục phổ thông.
  - `entities`: Các đối tượng tham gia.
  - `variables`: Danh sách ẩn số, đơn vị đo, điều kiện thực tế ($x > 0$, $x$ nguyên...).
  - `givens`: Các dữ kiện/giả thiết đã biết.
  - `relationships`: Quan hệ toán học giữa các đại lượng.
  - `target`: Câu hỏi/đích cần giải.

### 2.2. Symbolic Agent (Tác tử Ký hiệu Toán học)
- **Nhiệm vụ:** Tiếp nhận dữ kiện từ Parser Agent, ánh xạ quan hệ tự nhiên thành phương trình toán học hình thức và sinh mã Python bằng thư viện `sympy`.
- **Ràng buộc sinh mã:**
  - Khai báo ẩn qua `symbols('x ...', real=True, positive=True)`.
  - Thiết lập phương trình qua `Eq(lhs, rhs)`.
  - Phân số chính xác dùng `Rational(p, q)` tránh sai số dấu phẩy động.
  - Lọc nghiệm hợp lệ theo điều kiện thực tế (loại nghiệm âm, nghiệm vô lý).
  - Bắt buộc gán kết quả cuối cùng vào biến `result`.
  - Tuyệt đối chỉ sử dụng `sympy` và `math`.

### 2.3. Safe Sandbox Executor (Hộp cát Thực thi An toàn)
- **Cơ chế 2 lớp bảo vệ:**
  1. **Lớp 1 - Phân tích cú pháp tĩnh AST (Abstract Syntax Tree):**
     - Quét cây AST trước khi cho phép chạy.
     - Lọc bỏ và chặn đứng mọi lệnh nguy hiểm: cấm import `os`, `sys`, `subprocess`, `socket`, `shutil`...; cấm các hàm `eval()`, `exec()`, `open()`, `compile()`; cấm truy cập dunder attribute (`__subclasses__`, `__globals__`...).
     - Giới hạn số lượng nút AST $\le 1000$ và độ dài mã $\le 4000$ ký tự để chống DoS.
  2. **Lớp 2 - Thực thi cô lập đa tiến trình với Timeout 3 giây:**
     - Sử dụng `multiprocessing.Process` độc lập.
     - Nếu mã bị kẹt vòng lặp vô hạn (ví dụ `while True:`), tiến trình cha sẽ phát hiện hết giờ sau 3s và gọi `process.terminate()` / `process.kill()` ngay lập tức, giải phóng hoàn toàn CPU.
     - Không gian tên `globals` được thu hẹp tối đa chỉ chứa các hàm đại số an toàn của SymPy.

### 2.4. Self-Correction Agent (Tác tử Tự sửa lỗi)
- **Kích hoạt khi:** Sandbox ném ngoại lệ (`SyntaxError`, `ZeroDivisionError`, `TypeError`, `NameError`, `Timeout`, vi phạm AST) hoặc tập nghiệm thu được bị rỗng.
- **Quy trình suy luận phục hồi:**
  - Tiếp nhận đề bài + đoạn mã lỗi + thông điệp lỗi (error message) + vết ngăn xếp (traceback).
  - Phân tích nguyên nhân: do đặt nhầm tên biến, mâu thuẫn phương trình, hay chia cho 0.
  - Tái cấu trúc và sinh lại mã SymPy sạch sẽ.
  - Giới hạn: Tối đa 3 vòng lặp để tránh lặp vô hạn.

### 2.5. Pedagogical Agent (Tác tử Sư phạm)
- **Nhiệm vụ:** Nhận nghiệm số học đã được kiểm chứng tuyệt đối từ Sandbox, kết hợp thông tin phân tích để soạn thảo lời giải hoàn chỉnh theo chuẩn sư phạm SGK Toán THCS Việt Nam.
- **Cấu trúc lời giải 5 bước chuẩn mực:**
  - **Bước 1:** Chọn ẩn số và đặt điều kiện thích hợp cho ẩn.
  - **Bước 2:** Biểu diễn các đại lượng chưa biết theo ẩn và các đại lượng đã biết.
  - **Bước 3:** Lập phương trình (hoặc hệ phương trình).
  - **Bước 4:** Giải phương trình (hệ phương trình) từng bước chi tiết.
  - **Bước 5:** Đối chiếu nghiệm với điều kiện ban đầu và kết luận bài toán.
- **Trình bày công thức:** 100% định dạng $\LaTeX$ chuẩn đẹp (`$...$` và `$$...$$`).

### 2.6. Mathematical Visualization Engine (Bộ sinh Đồ thị Trực quan)
- **Tích hợp:** Trong giao diện Streamlit (`app.py`), tự động tạo đồ thị minh họa cho từng dạng bài toán:
  - **Toán chuyển động:** Đồ thị quan hệ Quãng đường - Thời gian $s(t) = v \cdot t$, so sánh đồ thị lúc đi/về hoặc 2 phương tiện ngược/cùng chiều.
  - **Hệ phương trình bậc nhất hai ẩn:** Vẽ 2 đường thẳng trên hệ trục tọa độ $Oxy$ cắt nhau tại điểm nghiệm $(x_0, y_0)$.
  - **Toán năng suất / công việc:** Biểu đồ lũy kế tiến độ công việc theo thời gian so với kế hoạch dự kiến.

### 2.7. LLM Backbone & Cơ chế Chuỗi phân tầng Fail-fast (Tiered Cascade Fallback)
- **Google Gemini Client:**
  - Tự động luân chuyển mô hình theo thứ tự ưu tiên: `gemini-3.5-flash` ➔ `gemini-flash-lite-latest` ➔ `gemini-3.1-flash-lite`.
  - Cơ chế **Fail-fast**: Khi gặp mã lỗi `503` (High Demand / Overloaded), `429` (Rate Limit) hoặc `404` (Model Not Found), hệ thống lập tức kích hoạt mô hình kế tiếp mà không phải chờ đợi thụ động.
  - Áp dụng đồng bộ trên cả tầng SDK `google-genai` lẫn tầng `REST API Fallback` trực tiếp.
- **OpenRouter Client (Mã nguồn mở Đám mây):**
  - Tích hợp chuẩn OpenAI Chat Completions API qua endpoint `https://openrouter.ai/api/v1/chat/completions`.
  - Hỗ trợ mô hình suy luận `deepseek/deepseek-r1:free` hoàn toàn miễn phí, tự động trích xuất `content` và `reasoning_content`.
  - Thay thế hoàn toàn Ollama cục bộ, loại bỏ gánh nặng cài đặt GPU/môi trường trên máy cá nhân.

---

## 3. Cấu trúc Trạng thái (AgentState)

```python
class AgentState(TypedDict, total=False):
    problem: str                                # Đề bài toán ban đầu
    parsed_info: Optional[Dict[str, Any]]       # Thông tin bóc tách từ Parser Agent
    sympy_code: Optional[str]                   # Mã SymPy do Symbolic Agent sinh ra
    execution_result: Optional[Dict[str, Any]]  # Kết quả chạy trong Sandbox
    correction_history: List[Dict[str, Any]]    # Lịch sử các lần tự sửa lỗi
    attempt_count: int                          # Số lần sửa lỗi hiện tại
    max_attempts: int                           # Giới hạn số lần sửa (mặc định: 3)
    final_solution: Optional[str]               # Lời giải sư phạm SGK kèm LaTeX
    status: str                                 # Trạng thái quy trình
    error_log: Optional[str]                    # Nhật ký lỗi nếu gặp sự cố
    agent_logs: List[Dict[str, Any]]            # Log chi tiết phục vụ hiển thị Streamlit
```

---

## 4. Chỉ số Đánh giá Định lượng (Benchmark Metrics)

Hệ thống được thiết kế để đo lường 4 chỉ số khoa học thực nghiệm:

$$\text{Accuracy} = \frac{N_{\text{chính xác}}}{N_{\text{tổng số bài}}} \times 100\%$$

$$\text{ESR (Execution Success Rate)} = \frac{N_{\text{mã thực thi không lỗi}}}{N_{\text{tổng số bài}}} \times 100\%$$

$$\text{SCRR (Self-Correction Recovery Rate)} = \frac{N_{\text{bài phục hồi thành công}}}{N_{\text{bài gặp lỗi ở lần 1}}} \times 100\%$$

$$\text{Average Latency} = \frac{\sum t_i}{N_{\text{tổng số bài}}} \text{ (giây/bài)}$$
