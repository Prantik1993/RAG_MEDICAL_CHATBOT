from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="rag-medical-chatbot",
    version="0.1.0",
    author="Prantik Bose",
    description="A Retrieval-Augmented Generation (RAG) medical chatbot using LangChain, Hugging Face and OpenAI.",
    long_description="RAG Medical Chatbot — a Retrieval-Augmented Generation app built with LangChain and OpenAI.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
