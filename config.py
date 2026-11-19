# -*- coding: utf-8 -*-
"""
Cấu hình tập trung cho Hệ thống Multi-Agent Giải toán THCS.
Quản lý các thông số LLM, Sandbox, Benchmark và đường dẫn dữ liệu.
"""

import os
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Cấu hình LLM Providers
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Cấu hình Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Mô hình Gemini mặc định và chuỗi Fallback Fail-fast
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]

# Cấu hình OpenRouter API (Mã nguồn mở trên đám mây, ví dụ DeepSeek-R1 miễn phí)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Cấu hình Sandbox Thực thi SymPy
SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT", "3"))
MAX_AST_NODES = int(os.getenv("MAX_AST_NODES", "1000"))
MAX_CODE_LENGTH = int(os.getenv("MAX_CODE_LENGTH", "4000"))

# Cấu hình Vòng lặp Tự sửa lỗi (Self-Correction)
MAX_CORRECTION_ATTEMPTS = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "3"))

# Cấu hình Bộ sinh dữ liệu & Benchmark
DEFAULT_BENCHMARK_DATASET = DATA_DIR / "thcs_problems.json"
GSM8K_VI_DATASET = DATA_DIR / "gsm8k_vi_sample.json"
