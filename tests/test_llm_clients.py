# -*- coding: utf-8 -*-
"""
Kiểm thử đơn vị cho tầng LLM Clients và LLMFactory:
1. OpenRouterClient (OpenAI Chat Completions protocol, header, reasoning_content, lỗi mạng).
2. GeminiClient với cơ chế chuỗi phân tầng Fail-fast (Tiered Cascade Fallback chống lỗi 503/429/404).
3. LLMFactory phân phối đúng các Client (gemini, openrouter, mock) và chặn provider đã loại bỏ (ollama).
"""

import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

import config
from src.llm.client import GeminiClient, OpenRouterClient, MockLLMClient
from src.llm.factory import LLMFactory


class TestOpenRouterClient(unittest.TestCase):
    """Kiểm thử OpenRouterClient kết nối DeepSeek-R1 / đám mây mã nguồn mở."""

    def test_missing_api_key_raises_value_error(self):
        """Ném ValueError nếu chưa cấu hình OPENROUTER_API_KEY."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            client = OpenRouterClient(api_key="")
            with self.assertRaises(ValueError) as ctx:
                client.generate("1 + 1 = ?")
            self.assertIn("Chưa cấu hình OPENROUTER_API_KEY", str(ctx.exception))

    def test_successful_response_with_content(self):
        """Trích xuất thành công nội dung từ choices[0].message.content."""
        fake_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Kết quả là 42"
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            client = OpenRouterClient(api_key="sk-test-key", model="deepseek/deepseek-r1:free")
            res = client.generate("Hỏi đáp", system_instruction="Bạn là gia sư")

            self.assertEqual(res, "Kết quả là 42")
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(req.headers.get("Authorization"), "Bearer sk-test-key")
            self.assertEqual(req.headers.get("Content-type"), "application/json")
            self.assertEqual(req.headers.get("Http-referer"), "https://github.com/mat3399_final-project")
            self.assertEqual(req.headers.get("X-title"), "THCS Math Multi-Agent Solver")

            payload = json.loads(req.data.decode("utf-8"))
            self.assertEqual(payload["model"], "deepseek/deepseek-r1:free")
            self.assertEqual(len(payload["messages"]), 2)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")

    def test_strip_think_tags_from_content(self):
        """Loại bỏ sạch sẽ các khối thẻ <think>...</think> khi mô hình suy luận DeepSeek-R1 trả về nháp."""
        fake_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nBước 1: Phân tích bài toán\nBước 2: Giải\n</think>\n\n```python\nresult = 42\n```"
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = OpenRouterClient(api_key="sk-test-key")
            res = client.generate("Giải bài toán")
            self.assertEqual(res, "```python\nresult = 42\n```")
            self.assertNotIn("<think>", res)
            self.assertNotIn("</think>", res)

    def test_fallback_to_think_content_when_only_think(self):
        """Nếu mô hình chỉ trả về duy nhất khối <think>, trích xuất nội dung bên trong thay vì rỗng."""
        fake_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nLời giải trực tiếp không có phần sau think\n</think>"
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = OpenRouterClient(api_key="sk-test-key")
            res = client.generate("Giải bài toán")
            self.assertEqual(res, "Lời giải trực tiếp không có phần sau think")

    def test_auto_read_api_key_from_env_on_generate(self):
        """Tự động đọc lại OPENROUTER_API_KEY từ biến môi trường khi gọi generate nếu lúc init để trống."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-late-injected-key"}):
            client = OpenRouterClient(api_key="")
            fake_response = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                res = client.generate("Hello")
                self.assertEqual(res, "OK")
                req = mock_urlopen.call_args[0][0]
                self.assertEqual(req.headers.get("Authorization"), "Bearer sk-late-injected-key")

    def test_fallback_to_reasoning_content(self):
        """Trích xuất reasoning_content khi content rỗng (định dạng DeepSeek-R1)."""
        fake_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Suy luận logic và đáp số là 100"
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = OpenRouterClient(api_key="sk-test-key")
            res = client.generate("Tính toán")
            self.assertEqual(res, "Suy luận logic và đáp số là 100")

    def test_http_error_handling(self):
        """Xử lý ngoại lệ HTTPError từ OpenRouter API."""
        fp = io.BytesIO(b'{"error": {"message": "Invalid API Key"}}')
        http_err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=fp
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            client = OpenRouterClient(api_key="invalid-key")
            with self.assertRaises(RuntimeError) as ctx:
                client.generate("Xin chào")
            self.assertIn("Lỗi OpenRouter API (HTTP 401)", str(ctx.exception))

    def test_url_error_handling(self):
        """Xử lý ngoại lệ URLError khi mất kết nối mạng."""
        url_err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            client = OpenRouterClient(api_key="sk-test-key")
            with self.assertRaises(ConnectionError) as ctx:
                client.generate("Xin chào")
            self.assertIn("Không thể kết nối tới OpenRouter", str(ctx.exception))


class TestGeminiClientFailFast(unittest.TestCase):
    """Kiểm thử cơ chế Tiered Cascade Fail-fast của GeminiClient."""

    def test_missing_api_key_raises_value_error(self):
        """Ném ValueError nếu chưa cấu hình GEMINI_API_KEY."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            client = GeminiClient(api_key="")
            with self.assertRaises(ValueError) as ctx:
                client.generate("1 + 1 = ?")
            self.assertIn("Chưa cấu hình GEMINI_API_KEY", str(ctx.exception))

    def test_candidate_models_order(self):
        """Danh sách candidate models phải bắt đầu bằng model chỉ định, kế tiếp là các fallback."""
        client = GeminiClient(api_key="test-key", model="gemini-3.5-flash")
        candidates = client._get_candidate_models()
        self.assertEqual(candidates[0], "gemini-3.5-flash")
        self.assertIn("gemini-flash-lite-latest", candidates)
        self.assertIn("gemini-3.1-flash-lite", candidates)

        # Nếu chỉ định model là Lite thì Lite đứng đầu và không bị trùng
        client_lite = GeminiClient(api_key="test-key", model="gemini-flash-lite-latest")
        candidates_lite = client_lite._get_candidate_models()
        self.assertEqual(candidates_lite[0], "gemini-flash-lite-latest")
        self.assertEqual(len(candidates_lite), len(set(candidates_lite)))

    def test_gemini_auto_read_api_key_from_env_on_generate(self):
        """Tự động nhận diện GEMINI_API_KEY từ biến môi trường khi gọi generate nếu lúc khởi tạo để trống."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaLateInjectedKey"}):
            client = GeminiClient(api_key="")
            client._genai_client = None
            with patch.object(client, "_generate_via_rest", return_value="Phản hồi muộn"):
                res = client.generate("1 + 1")
                self.assertEqual(res, "Phản hồi muộn")
                self.assertEqual(client.api_key, "AIzaLateInjectedKey")

    def test_rest_fail_fast_on_500_502_504(self):
        """Mô hình đầu tiên gặp 500 / 502 / 504 gateway error cũng lập tức chuyển sang mô hình tiếp theo."""
        client = GeminiClient(api_key="test-key", model="gemini-3.5-flash")
        client._genai_client = None

        calls = []

        def fake_generate_via_rest(prompt, system_instruction=None, temperature=0.2, model_name=None):
            calls.append(model_name)
            if model_name == "gemini-3.5-flash":
                fp = io.BytesIO(b'{"error": "502 Bad Gateway"}')
                raise urllib.error.HTTPError("url", 502, "Bad Gateway", {}, fp)
            elif model_name == "gemini-flash-lite-latest":
                return "Phản hồi từ Lite sau khi 3.5 bị 502!"
            return "Khác"

        with patch.object(client, "_generate_via_rest", side_effect=fake_generate_via_rest):
            res = client.generate("Giải phương trình")
            self.assertEqual(res, "Phản hồi từ Lite sau khi 3.5 bị 502!")
            self.assertEqual(calls, ["gemini-3.5-flash", "gemini-flash-lite-latest"])

    def test_rest_fail_fast_on_503_and_429(self):
        """Mô hình đầu tiên gặp 503 / 429 lập tức chuyển sang mô hình tiếp theo mà không dừng luồng."""
        client = GeminiClient(api_key="test-key", model="gemini-3.5-flash")
        # Đảm bảo tắt SDK client để chạy qua REST
        client._genai_client = None

        calls = []

        def fake_generate_via_rest(prompt, system_instruction=None, temperature=0.2, model_name=None):
            calls.append(model_name)
            if model_name == "gemini-3.5-flash":
                fp = io.BytesIO(b'{"error": "503 High demand Overloaded"}')
                raise urllib.error.HTTPError("url", 503, "Service Unavailable", {}, fp)
            elif model_name == "gemini-flash-lite-latest":
                return "Phản hồi thành công từ Lite!"
            return "Khác"

        with patch.object(client, "_generate_via_rest", side_effect=fake_generate_via_rest):
            res = client.generate("Giải phương trình")
            self.assertEqual(res, "Phản hồi thành công từ Lite!")
            self.assertEqual(calls, ["gemini-3.5-flash", "gemini-flash-lite-latest"])

    def test_sdk_fail_fast_on_overloaded_error(self):
        """Tầng SDK gặp lỗi overloaded/503 lập tức kích hoạt Fail-fast sang model kế tiếp."""
        client = GeminiClient(api_key="test-key", model="gemini-3.5-flash")
        mock_sdk = MagicMock()
        client._genai_client = mock_sdk
        client._genai_types = MagicMock()

        sdk_calls = []

        def fake_sdk_generate(model, contents, config=None):
            sdk_calls.append(model)
            if model == "gemini-3.5-flash":
                raise Exception("503 Resource Unavailable: High demand on gemini-3.5-flash")
            mock_res = MagicMock()
            mock_res.text = "Phản hồi SDK từ " + model
            return mock_res

        mock_sdk.models.generate_content.side_effect = fake_sdk_generate

        res = client.generate("1 + 1")
        self.assertEqual(res, "Phản hồi SDK từ gemini-flash-lite-latest")
        self.assertEqual(sdk_calls, ["gemini-3.5-flash", "gemini-flash-lite-latest"])


class TestLLMFactory(unittest.TestCase):
    """Kiểm thử LLMFactory khởi tạo đúng lớp theo provider."""

    def test_create_gemini_client(self):
        client = LLMFactory.create_client(provider="gemini", api_key="test-key", model="gemini-3.5-flash")
        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(client.model, "gemini-3.5-flash")

    def test_create_openrouter_client(self):
        client = LLMFactory.create_client(provider="openrouter", api_key="sk-or-test", model="deepseek/deepseek-r1:free")
        self.assertIsInstance(client, OpenRouterClient)
        self.assertEqual(client.model, "deepseek/deepseek-r1:free")

    def test_create_mock_client(self):
        client = LLMFactory.create_client(provider="mock")
        self.assertIsInstance(client, MockLLMClient)

    def test_ollama_provider_unsupported_raises_error(self):
        """Ollama đã bị gỡ bỏ nên khi truyền provider='ollama' sẽ ném ValueError."""
        with self.assertRaises(ValueError) as ctx:
            LLMFactory.create_client(provider="ollama")
        self.assertIn("Nhà cung cấp LLM không được hỗ trợ", str(ctx.exception))
        self.assertIn("Chỉ hỗ trợ: 'gemini', 'openrouter', 'mock'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
