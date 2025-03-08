"""
LLM provider integrations for GeoGPT
"""

import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def get_llm(provider: Optional[str] = None):
    """
    Returns an LLM instance based on the provider.

    Args:
        provider (str, optional): The LLM provider to use.
            Options: "openai", "google", "anthropic", "deepseek"
            If None, will use the LLM_PROVIDER environment variable,
            or default to "openai".

    Returns:
        LLM: A LangChain LLM instance
    """
    # Load environment variables from .env if it exists
    # This is a backup and users should set environment variables directly
    # when using the package in production
    try:
        load_dotenv()  # Will look for .env in the current directory
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")

    # If no provider specified, get from environment or default to OpenAI
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")

    logger.info(f"Initializing LLM with provider: {provider}")

    if provider == "google":
        MODEL = os.getenv("LLM_MODEL_GOOGLE", "gemini-2.0-flash-exp")
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No GOOGLE_API_KEY found in environment variables")

        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=0.0,
        )
        logger.info(f"Initialized Google LLM with model: {MODEL}")

    elif provider == "anthropic":
        MODEL = os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-7-sonnet-latest")
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")

        llm = ChatAnthropic(
            model=MODEL,
            temperature=0.0,
        )
        logger.info(f"Initialized Anthropic LLM with model: {MODEL}")

    elif provider == "deepseek":
        MODEL = os.getenv("LLM_MODEL_DEEPSEEK", "deepseek-chat")
        # Check for API key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("No DEEPSEEK_API_KEY found in environment variables")

        llm = ChatDeepSeek(
            model=MODEL,
            temperature=0.0,
        )
        logger.info(f"Initialized DeepSeek LLM with model: {MODEL}")

    elif provider == "openai":  # Default to OpenAI
        MODEL = os.getenv("LLM_MODEL_OPENAI", "gpt-4o")
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY found in environment variables")

        llm = ChatOpenAI(
            model=MODEL,
            temperature=0.0,
        )
        logger.info(f"Initialized OpenAI LLM with model: {MODEL}")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return llm
