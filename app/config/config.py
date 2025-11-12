import os

# Base directory of your project
BASE_DIR = os.getcwd()

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Local paths (with environment overrides)
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "data"))
DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", os.path.join(BASE_DIR, "vectorstore", "db_faiss"))

# Chunking parameters
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
