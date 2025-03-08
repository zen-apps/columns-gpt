"""
LLM integration utilities for ColumnsGPT.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def return_llm(provider="openai"):
    """Initialize and return an LLM based on the specified provider.
    
    Args:
        provider (str): The LLM provider to use. Options: "openai", "google", "anthropic", "deepseek".
            Defaults to "openai".
            
    Returns:
        langchain.chat_models.ChatModel: A configured LLM instance
    """
    if provider == "google":
        MODEL = os.getenv("LLM_MODEL_GOOGLE", "gemini-2.0-flash-exp")  # Default model
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", MODEL),
            temperature=0.0,
        )
    elif provider == "anthropic":
        MODEL = os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-5-sonnet-latest")
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", MODEL),
            temperature=0.0,
        )
    elif provider == "deepseek":
        MODEL = os.getenv("LLM_MODEL_DEEPSEEK", "deepseek-chat")  # Default model
        llm = ChatDeepSeek(
            model=MODEL,
            temperature=0.0,
        )
    elif provider == "openai":  # Default to OpenAI
        MODEL = os.getenv("LLM_MODEL_OPENAI", "gpt-4o")  # Default model
        llm = ChatOpenAI(
            model=MODEL,
            temperature=0.0,
        )
    return llm