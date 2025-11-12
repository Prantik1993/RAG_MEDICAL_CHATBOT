import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)


def load_vector_store():
    try:
        embedding_model = get_embedding_model()

        # Ensure vector store directory exists
        if not os.path.isdir(DB_FAISS_PATH):
            raise CustomException(
                f"Vector store not found at path: {DB_FAISS_PATH}. "
                f"Please run data_loader.py to generate embeddings."
            )

        logger.info(f"Loading FAISS vector store from: {DB_FAISS_PATH}")

        # Load FAISS database (safer without dangerous deserialization)
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info("FAISS vector store loaded successfully.")
        return db

    except Exception as e:
        error_message = CustomException("Failed to load FAISS vector store", e)
        logger.error(str(error_message))
        raise error_message


def save_vector_store(text_chunks: list):
    try:
        if not text_chunks:
            raise CustomException("No text chunks were provided for vectorization.")

        logger.info("Starting FAISS vector store creation...")

        # Ensure the directory exists before saving
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        logger.info(f"Ensured vector store directory exists: {DB_FAISS_PATH}")

        # Load embedding model and create FAISS vector database
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)

        # Save locally for reuse
        db.save_local(DB_FAISS_PATH)
        logger.info(f"FAISS vector store saved successfully at: {DB_FAISS_PATH}")

        return db

    except Exception as e:
        error_message = CustomException("Failed to create or save FAISS vector store", e)
        logger.error(str(error_message))
        raise error_message
