import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)

def load_llm(model_name: str = None, api_key: str = None):
    """Loads the OpenAI LLM using environment variables."""
    try:
        model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise CustomException("OPENAI_API_KEY not set in .env")

        logger.info(f"Loading OpenAI model → {model_name}")

        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.2,
            max_tokens=800
        )

        logger.info(f"✅ OpenAI LLM loaded successfully: {model_name}")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load OpenAI LLM", e)
        logger.error(str(error_message))
        return None
