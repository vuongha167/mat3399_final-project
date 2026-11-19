# -*- coding: utf-8 -*-
"""
Factory tạo đối tượng LLM Client dựa trên cấu hình nhà cung cấp (Gemini, OpenRouter, Mock).
"""

from typing import Optional
import config
from src.llm.client import BaseLLMClient, GeminiClient, OpenRouterClient, MockLLMClient


class LLMFactory:
    """Factory Pattern quản lý việc tạo đối tượng LLM Client."""

    @staticmethod
    def create_client(
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Khởi tạo LLM Client tương ứng.

        Args:
            provider: "gemini", "openrouter", hoặc "mock" (mặc định lấy từ config).
            api_key: Khóa API cho Gemini hoặc OpenRouter.
            model: Tên mô hình (ví dụ "gemini-3.5-flash" hoặc "deepseek/deepseek-r1:free").
            base_url: Base URL cho OpenRouter (nếu dùng endpoint tùy biến).

        Returns:
            Thể hiện của BaseLLMClient.
        """
        selected_provider = (provider or config.DEFAULT_PROVIDER).lower()

        if selected_provider == "gemini":
            key = api_key or config.GEMINI_API_KEY
            selected_model = model or config.DEFAULT_GEMINI_MODEL
            return GeminiClient(api_key=key, model=selected_model)

        elif selected_provider == "openrouter":
            key = api_key or config.OPENROUTER_API_KEY
            selected_model = model or config.DEFAULT_OPENROUTER_MODEL
            url = base_url or config.OPENROUTER_BASE_URL
            return OpenRouterClient(api_key=key, model=selected_model, base_url=url)

        elif selected_provider == "mock":
            return MockLLMClient(**kwargs)

        else:
            raise ValueError(
                f"Nhà cung cấp LLM không được hỗ trợ: '{selected_provider}'. "
                f"Chỉ hỗ trợ: 'gemini', 'openrouter', 'mock'."
            )
