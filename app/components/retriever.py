from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical assistant. 
Use only the provided context to answer the question clearly and accurately.
If the answer is not in the context, say "I'm sorry, I couldn't find that information in the provided material."

Context:
{context}
"""

def create_qa_chain():

    try:
        logger.info("🔍 Loading vector store...")
        db = load_vector_store()
        if db is None:
            raise CustomException("VectorStore could not be loaded.")

        logger.info("🤖 Loading OpenAI LLM...")
        llm = load_llm()
        if llm is None:
            raise CustomException("LLM could not be loaded.")

        # Prompt construction
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}")
        ])

        # Chain that merges retrieved docs + prompt + LLM
        combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        # Retrieval pipeline
        retrieval_chain = create_retrieval_chain(
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain=combine_chain
        )

        logger.info("✅ QA Retrieval chain created successfully (LangChain v1.x)")
        return retrieval_chain

    except Exception as e:
        error_message = CustomException("❌ Failed to create QA chain", e)
        logger.error(str(error_message))
        return None
