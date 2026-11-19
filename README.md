# 📐 Đề tài số 4: Ứng dụng Multi-Agent Framework kết hợp Tool Use (SymPy/Calculator) cho suy luận số đa bước trong chương trình Toán THCS

Hệ thống AI Tutor giải toán THCS thông minh tích hợp **Kiểm chứng Ký hiệu (Symbolic Verification)**, kết hợp điều phối đa tác tử qua `LangGraph` và môi trường thực thi an toàn `SymPy Sandbox`.

---

## 🌟 Điểm nổi bật của Dự án

1. **Kiểm chứng Ký hiệu Chính xác Tuyệt đối:** Khắc phục triệt để hiện tượng mô hình ngôn ngữ lớn (LLM) bị ảo giác khi thực hiện tính toán số học đa bước phức tạp bằng cách chuyển đổi bài toán thành biểu thức đại số và giao quyền giải cho thư viện `SymPy`.
2. **Quy trình Tuần hoàn 5 Tác tử Chuyên biệt:**
   - 🔍 **Parser Agent:** Bóc tách đề bài tự nhiên tiếng Việt thành cấu trúc dữ liệu toán học (ẩn số, điều kiện, giả thiết).
   - ⚙️ **Symbolic Agent:** Thiết lập hệ phương trình và sinh mã thực thi Python SymPy.
   - 🛡️ **Safe Sandbox Executor:** Phân tích cú pháp trừu tượng AST (chặn đứng `os`, `sys`, `subprocess`, `eval`...) và khống chế thời gian thực thi (Timeout 3 giây bằng multiprocessing).
   - 🔄 **Self-Correction Agent:** Tự động bắt lỗi runtime, phân tích nguyên nhân và sửa đổi mã (tối đa 3 vòng lặp).
   - 🎓 **Pedagogical Agent:** Trình bày lời giải sư phạm chuẩn 5 bước theo sách giáo khoa Toán THCS Việt Nam kèm công thức $\LaTeX$ sắc nét.
3. **Mô hình Đám mây Linh hoạt & Đa dạng:** Hỗ trợ **Google Gemini** thế hệ mới với cơ chế chuỗi phân tầng **Fail-fast** (tự động luân chuyển `gemini-3.5-flash` ➔ `gemini-flash-lite-latest` ➔ `gemini-3.1-flash-lite` khi gặp 503/429) và **OpenRouter** kết nối trực tiếp mô hình suy luận mã nguồn mở **DeepSeek-R1** hoàn toàn miễn phí mà không cần cài đặt cục bộ.
4. **Sản phẩm Đầy đủ & Hoàn chỉnh:**
   - 🌐 Giao diện Web tương tác trực quan bằng **Streamlit** (`app.py`), hiển thị tiến trình đa tác tử theo thời gian thực và trực quan hóa **đồ thị toán học** (đồ thị chuyển động $s(t)$, hệ đường thẳng giao nhau, tiến độ năng suất).
   - 📊 Bộ công cụ đo lường thực nghiệm định lượng **Benchmark** (`benchmark.py`) hỗ trợ cả dữ liệu cục bộ và bộ dữ liệu chuẩn mực **`hllj/vi_gsm8k`** (1.319 câu) từ Hugging Face.
   - 📓 Sẵn sàng chạy 1 chạm trên **Google Colab** (`notebook_colab.ipynb`) tích hợp xem trước dữ liệu pandas, chạy benchmark và mở đường hầm **Localtunnel** / **Ngrok**.
   - 🧪 Hệ thống 54 bài kiểm thử đơn vị tự động toàn diện (`tests/`).

---

## 📂 Cấu trúc Thư mục Dự án

```
mat3399_final-project/
├── AGENTS.md                   # Đặc tả kiến trúc Multi-Agent & Vòng lặp Self-Correction
├── README.md                   # Hướng dẫn sử dụng toàn diện
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── notebook_colab.ipynb        # Jupyter Notebook chạy 1 chạm trên Google Colab
├── app.py                      # Ứng dụng Web tương tác Streamlit
├── benchmark.py                # Script đo lường định lượng (Accuracy, ESR, SCRR)
├── config.py                   # Cấu hình hệ thống tập trung
├── src/
│   ├── llm/                    # Quản lý LLM Client & Factory (Gemini, OpenRouter, Mock)
│   │   ├── client.py
│   │   └── factory.py
│   ├── agents/                 # Các tác tử thông minh
│   │   ├── state.py            # TypedDict cấu trúc trạng thái luồng
│   │   ├── parser.py           # Parser Agent
│   │   ├── symbolic.py         # Symbolic Agent
│   │   ├── corrector.py        # Self-Correction Agent
│   │   └── pedagogical.py      # Pedagogical Agent
│   ├── sandbox/                # Hộp cát thực thi an toàn
│   │   ├── ast_validator.py    # Phân tích cây AST lọc lệnh nguy hiểm
│   │   ├── executor.py         # Thực thi mã có timeout 3s & bắt ngoại lệ
│   │   └── fallback_sympy.py   # Bộ ký hiệu toán học dự phòng thuần Python
│   └── workflow.py             # Điều phối quy trình Multi-Agent (LangGraph & Pure Runner)
├── data/
│   ├── thcs_problems.json      # 15 bài toán THCS kinh điển (chuyển động, năng suất, nồng độ...)
│   └── gsm8k_vi_sample.json    # 10 bài toán GSM8K Việt hóa suy luận số đa bước
└── tests/
    ├── test_ast_validator.py   # Unit test AST Sandbox bảo mật
    ├── test_sandbox_executor.py# Unit test Sandbox timeout & execution
    ├── test_workflow_mock.py   # Test tích hợp luồng Multi-Agent với Mock LLM
    ├── test_benchmark.py       # Test trích xuất đáp án GSM8K và logic benchmark
    └── test_llm_clients.py     # Test OpenRouter, Gemini Fail-fast 3.x và LLMFactory
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy trên Máy tính (Local)

### 1. Chuẩn bị Môi trường ảo Python
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
python3 -m venv venv
source venv/bin/activate    # Trên Linux/macOS
# venv\Scripts\activate     # Trên Windows
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Cấu hình Khóa API
Hệ thống hỗ trợ 2 nguồn cung cấp LLM chính:
1. **Google Gemini** (khuyến nghị mặc định `gemini-3.5-flash` kèm Fail-fast sang Lite):
```bash
export GEMINI_API_KEY="AIzaSy..."    # Lấy tại https://aistudio.google.com/
```

2. **OpenRouter (Mã nguồn mở Đám mây)** (hỗ trợ `deepseek/deepseek-r1:free` miễn phí 100%):
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."  # Lấy tại https://openrouter.ai/keys
```
*(Cả hai khóa API đều có thể nhập trực tiếp hoặc tự động nhận diện từ môi trường trên giao diện web Streamlit).*

---

## 💻 Hướng dẫn Sử dụng Chi tiết

### 1. Khởi chạy Ứng dụng Web Tương tác Streamlit
```bash
streamlit run app.py
```
Sau khi khởi chạy, truy cập trình duyệt tại: `http://localhost:8501`.
- Chọn mô hình (Gemini / OpenRouter / Mock).
- Chọn bài toán mẫu từ thư viện hoặc tự nhập bài toán THCS bằng tiếng Việt.
- Bấm nút **"Bắt đầu Giải toán"** và theo dõi tiến trình thực thi thời gian thực của 5 Tác tử.

### 2. Chạy Script Đánh giá Hiệu năng Định lượng (Benchmark)

Hệ thống hỗ trợ các chế độ dữ liệu kiểm nghiệm linh hoạt:
- **`--dataset local`** (Mặc định): Đánh giá trên tập bài toán THCS tuyển chọn (`data/thcs_problems.json`).
- **`--dataset gsm8k`**: Đánh giá trên tập 10 bài GSM8K Việt hóa nội bộ (`data/gsm8k_vi_sample.json`).
- **`--dataset vi_gsm8k`**: Tải trực tiếp **1.319 câu hỏi** từ bộ dữ liệu chuẩn Hugging Face `hllj/vi_gsm8k` (split `test`) vào RAM, tự động trích xuất đáp án sau `####` và so khớp với nghiệm giải được của SymPy.

```bash
# 1. Chạy trên tập THCS nội bộ với Gemini (tự động Fail-fast 3.5 -> Lite khi gặp 503)
python3 benchmark.py --dataset local --provider gemini --model gemini-3.5-flash

# 2. Chạy trên tập THCS nội bộ với OpenRouter DeepSeek-R1 (miễn phí)
python3 benchmark.py --dataset local --provider openrouter --model deepseek/deepseek-r1:free

# 3. Chạy trên tập 10 bài GSM8K Việt hóa nội bộ
python3 benchmark.py --dataset gsm8k --provider gemini

# 4. Chạy trên bộ dữ liệu chuẩn Hugging Face vi_gsm8k (chạy thử nghiệm 10 câu đầu)
python3 benchmark.py --dataset vi_gsm8k --provider gemini --limit 10 --delay 1.0

# 5. Chạy toàn bộ 1.319 câu hỏi của tập test vi_gsm8k
python3 benchmark.py --dataset vi_gsm8k --provider gemini --delay 0.5

# 6. Chạy thử nghiệm nhanh không tốn API key (chế độ Mock)
python3 benchmark.py --dataset local --provider mock --limit 5
```

Kết quả báo cáo định lượng sẽ được in ra bảng điều khiển và tự động xuất ra tệp `benchmark_results.json`:
- **Độ chính xác (Accuracy %)**
- **Tỷ lệ thực thi thành công của mã ký hiệu (ESR %)**
- **Tỷ lệ tự sửa lỗi phục hồi thành công (SCRR %)**
- **Thời gian phản hồi trung bình (Average Latency)**

### 3. Chạy Toàn bộ Bộ Kiểm thử Tự động (Unit Tests)
```bash
python3 -m unittest discover tests
```
*Tất cả 54 bài test đơn vị đều chạy độc lập, siêu tốc (<1 giây) và bảo đảm mã nguồn an toàn tuyệt đối.*

---

## ☁️ Hướng dẫn Chạy 1 chạm trên Google Colab

Dự án cung cấp tệp notebook chuyên dụng [`notebook_colab.ipynb`](notebook_colab.ipynb) được tối ưu sẵn:
1. Mở [Google Colab](https://colab.research.google.com/) và tải tệp `notebook_colab.ipynb` lên (hoặc clone repository).
2. Chạy tuần tự các bước:
   - **Bước 1:** Chuẩn bị môi trường & cài đặt tự động dependencies (`datasets`, `sympy`, `langgraph`...).
   - **Bước 2:** Cấu hình `GEMINI_API_KEY` và `OPENROUTER_API_KEY` (hỗ trợ nhập an toàn qua `google.colab.userdata` hoặc `getpass`).
   - **Bước 2.1:** **Khám phá Dữ liệu Chuẩn GSM8K Tiếng Việt (`hllj/vi_gsm8k`)** - xem trước 5 mẫu đầu tiên gồm đề bài, diễn giải và đáp án số học ra bảng `pandas` tương tác.
   - **Bước 3:** Thử nghiệm giải bài toán THCS 1 chạm với Multi-Agent Solver (mặc định dùng Gemini 3.5 Flash tự động Fail-fast).
   - **Bước 4:** Chạy Benchmark đo lường định lượng toàn diện:
     - **Bước 4.1:** Benchmark trên tập toán THCS nội bộ với Gemini Fail-fast (`--dataset local --provider gemini`).
     - **Bước 4.2:** Benchmark trên tập toán THCS nội bộ với OpenRouter DeepSeek-R1 (`--dataset local --provider openrouter`).
     - **Bước 4.3:** Benchmark trên tập chuẩn quốc tế Hugging Face (`--dataset vi_gsm8k`).
   - **Bước 5:** Khởi chạy Streamlit kèm đường hầm truy cập công khai `localtunnel` hoặc `ngrok` để mở giao diện Web từ bất kỳ đâu!

---

## 📊 Bộ Dữ liệu Mẫu (Benchmark Datasets)

Dự án hỗ trợ 3 nguồn dữ liệu kiểm nghiệm:
1. **`data/thcs_problems.json`**: 15 bài toán đại số THCS kinh điển được tuyển chọn từ các đề thi vào 10 và SGK Toán 6, 7, 8, 9 (Toán chuyển động xuôi/ngược dòng, năng suất làm chung/riêng, nồng độ dung dịch muối, bài toán kinh tế lãi suất, hình học thực tế, tìm số).
2. **`data/gsm8k_vi_sample.json`**: 10 bài toán suy luận số đa bước trích xuất từ tập GSM8K được dịch sang tiếng Việt tự nhiên và điều chỉnh số liệu phù hợp với tâm lý lứa tuổi học sinh THCS.
3. **`Hugging Face: hllj/vi_gsm8k`**: Toàn bộ **1.319 bài toán** suy luận số học đa bước trong tập `test` của bộ dữ liệu GSM8K Việt hóa chuẩn mực. Được tải trực tiếp vào RAM thông qua thư viện `datasets` mà không cần lưu tệp đĩa trung gian, giúp đánh giá định lượng trên quy mô lớn đạt độ tin cậy khoa học cao nhất.

---

## ⚖️ Giấy phép & Tác giả
- Đề tài số 4: Nghiên cứu AI Tutor & Multi-Agent Reasoning.
- Phục vụ nghiên cứu khoa học, đồ án môn học và ứng dụng giáo dục thông minh.
