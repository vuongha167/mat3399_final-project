# -*- coding: utf-8 -*-
"""
Unit tests cho Hệ thống Benchmark Runner và Trích xuất Dữ liệu GSM8K.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from benchmark import extract_gsm8k_answer, is_answer_close, load_benchmark_data
import config


class TestBenchmarkUtils(unittest.TestCase):
    """Kiểm thử các hàm tiện ích và trích xuất đáp án benchmark."""

    def test_extract_gsm8k_answer_integers(self):
        """Kiểm tra trích xuất số nguyên chuẩn sau ####."""
        text = "Nam có tất cả 105 viên bi.\n#### 105"
        self.assertEqual(extract_gsm8k_answer(text), 105)

    def test_extract_gsm8k_answer_with_commas(self):
        """Kiểm tra trích xuất số có dấu phẩy phân tách hàng nghìn."""
        text = "Tổng số tiền thu được là 1,200 nghìn đồng.\n#### 1,200"
        self.assertEqual(extract_gsm8k_answer(text), 1200)

    def test_extract_gsm8k_answer_negative_and_decimal(self):
        """Kiểm tra trích xuất số âm và số thập phân."""
        self.assertEqual(extract_gsm8k_answer("#### -42"), -42)
        self.assertEqual(extract_gsm8k_answer("#### 3.5"), 3.5)

    def test_extract_gsm8k_answer_fraction_or_text(self):
        """Kiểm tra trích xuất phân số dạng chuỗi nếu không ép kiểu được."""
        self.assertEqual(extract_gsm8k_answer("#### 3/4"), "3/4")
        self.assertIsNone(extract_gsm8k_answer(None))

    def test_extract_gsm8k_answer_currency_and_units(self):
        """Kiểm tra trích xuất đáp án có kèm ký hiệu tiền tệ, đơn vị hoặc phần trăm."""
        self.assertEqual(extract_gsm8k_answer("#### $1,200"), 1200)
        self.assertEqual(extract_gsm8k_answer("#### 105 viên bi"), 105)
        self.assertEqual(extract_gsm8k_answer("#### 25%"), 25)
        self.assertEqual(extract_gsm8k_answer("#### -15.5 kg"), -15.5)

    def test_is_answer_close_scalar(self):
        """Kiểm tra so sánh nghiệm đơn lẻ."""
        self.assertTrue(is_answer_close(50, 50))
        self.assertTrue(is_answer_close("50", 50))
        self.assertTrue(is_answer_close(50.00001, 50))
        self.assertFalse(is_answer_close(50, 60))

    def test_is_answer_close_fractions_and_currency(self):
        """Kiểm tra so sánh tương đương giữa phân số chuỗi, số thực và ký hiệu tiền tệ."""
        self.assertTrue(is_answer_close(0.75, "3/4"))
        self.assertTrue(is_answer_close("3/4", 0.75))
        self.assertTrue(is_answer_close(1200, "$1,200"))
        self.assertTrue(is_answer_close(105, "105 viên bi"))
        self.assertTrue(is_answer_close(25, "25%"))

    def test_is_answer_close_containers(self):
        """Kiểm tra so sánh nghiệm trong list hoặc dict."""
        self.assertTrue(is_answer_close([20, 50], 50))
        self.assertTrue(is_answer_close({"x": 50, "y": 20}, 50))
        self.assertFalse(is_answer_close([10, 20], 50))


class TestBenchmarkDatasetLoading(unittest.TestCase):
    """Kiểm thử cơ chế nạp bộ dữ liệu benchmark (local và vi_gsm8k)."""

    def test_load_local_default_dataset(self):
        """Kiểm tra tải tập dữ liệu local mặc định."""
        problems, desc = load_benchmark_data(dataset_mode="local")
        self.assertGreater(len(problems), 0)
        self.assertIn("thcs_problems.json", desc)
        first = problems[0]
        self.assertIn("question", first)
        self.assertIn("expected_answer", first)

    def test_load_local_custom_file(self):
        """Kiểm tra tải tập dữ liệu qua đường dẫn JSON tùy chọn."""
        sample_path = config.DATA_DIR / "gsm8k_vi_sample.json"
        problems, desc = load_benchmark_data(dataset_mode="local", dataset_path=sample_path)
        self.assertEqual(len(problems), 10)
        self.assertIn("gsm8k_vi_sample.json", desc)

    def test_load_local_gsm8k_alias(self):
        """Kiểm tra tải tập dữ liệu gsm8k mẫu qua alias trực tiếp."""
        problems, desc = load_benchmark_data(dataset_mode="gsm8k")
        self.assertEqual(len(problems), 10)
        self.assertIn("gsm8k_vi_sample.json", desc)

    def test_load_custom_json_with_answer_field(self):
        """Kiểm tra tự động chuyển đổi trường answer thành expected_answer cho file JSON bên ngoài."""
        import tempfile
        import json
        custom_data = [
            {"id": "c1", "question": "Hỏi Nam?", "answer": "Giải thích...\n#### 99"}
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(custom_data, f)
            temp_name = f.name

        try:
            problems, desc = load_benchmark_data(dataset_mode=temp_name)
            self.assertEqual(len(problems), 1)
            self.assertEqual(problems[0]["expected_answer"], 99)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def test_load_invalid_file_exits(self):
        """Kiểm tra báo lỗi và thoát khi đường dẫn tệp không tồn tại."""
        with self.assertRaises(SystemExit):
            load_benchmark_data(dataset_mode="non_existent_file.json")

    def test_load_vi_gsm8k_missing_datasets_fallback(self):
        """Kiểm tra cơ chế graceful fallback khi môi trường thiếu thư viện datasets."""
        with patch.dict(sys.modules, {"datasets": None}):
            # Khi import datasets ném ModuleNotFoundError
            with patch("builtins.__import__", side_effect=ModuleNotFoundError("No module named 'datasets'")):
                with self.assertRaises(SystemExit):
                    load_benchmark_data(dataset_mode="vi_gsm8k")

    def test_load_vi_gsm8k_with_mocked_datasets(self):
        """Kiểm tra tải và chuẩn hóa dữ liệu vi_gsm8k khi thư viện datasets sẵn sàng."""
        mock_sample_dataset = [
            {"question": "Bài toán 1", "answer": "Giải thích 1\n#### 42"},
            {"question": "Bài toán 2", "answer": "Giải thích 2\n#### 1,500"}
        ]
        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = mock_sample_dataset

        with patch.dict(sys.modules, {"datasets": mock_datasets_module}):
            problems, desc = load_benchmark_data(dataset_mode="vi_gsm8k")
            self.assertEqual(len(problems), 2)
            self.assertIn("hllj/vi_gsm8k", desc)
            self.assertEqual(problems[0]["expected_answer"], 42)
            self.assertEqual(problems[1]["expected_answer"], 1500)
            self.assertEqual(problems[0]["category"], "GSM8K Tiếng Việt")


if __name__ == "__main__":
    unittest.main()
