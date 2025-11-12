# from app.common.custom_exception import CustomException

# try:
#     result = 10/-0
# except Exception as e:
#     raise CustomException("Divison",e)


"""
Test script to verify that the entire RAG pipeline works end-to-end.
It checks FAISS, LLM, and QA retrieval in one run.
"""

from app.components.data_loader import process_and_store_pdfs
from app.components.vector_store import load_vector_store
from app.components.llm import load_llm
from app.components.retriever import create_qa_chain
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def test_full_pipeline():
    try:
        logger.info("🚀 Starting RAG pipeline test...")

        # STEP 1️⃣: Ensure FAISS vector store exists
        try:
            db = load_vector_store()
            if db is not None:
                logger.info("✅ Existing vector store loaded successfully.")
            else:
                logger.warning("⚠️ Vector store not found — building a new one.")
                db = process_and_store_pdfs()
        except Exception as e:
            raise CustomException("Vector store test failed", e)

        # STEP 2️⃣: Load LLM
        try:
            llm = load_llm()
            if llm is None:
                raise CustomException("LLM could not be initialized.")
            logger.info("✅ LLM initialized successfully.")
        except Exception as e:
            raise CustomException("LLM test failed", e)

        # STEP 3️⃣: Create QA chain
        try:
            qa_chain = create_qa_chain()
            if qa_chain is None:
                raise CustomException("QA chain creation failed.")
            logger.info("✅ QA chain created successfully.")
        except Exception as e:
            raise CustomException("QA chain test failed", e)

        # STEP 4️⃣: Run a sample query
        sample_query = "what is cancer"
        logger.info(f"🧠 Running sample query: {sample_query}")

        result = qa_chain.invoke({"input": sample_query})
        answer = result.get("output_text", result)

        print("\n===============================")
        print("✅ RAG Pipeline Test Result")
        print("===============================")
        print(f"Sample Question: {sample_query}")
        print(f"Answer: {answer}")
        print("===============================\n")

        logger.info("🎉 Full RAG pipeline test completed successfully.")
        return True

    except Exception as e:
        logger.error(f"❌ RAG pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("✅ Everything is working end-to-end!")
    else:
        print("❌ Some part of the pipeline failed — check logs.")


