from setuptools import setup, find_packages

setup(
    name="rag_project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "llama-index",
        "transformers",
        "torch",
        "requests",
        "bs4",
        "pytesseract",
        "opencv-python",
        "pydub",
        "speechrecognition",
        "fastapi",
        "uvicorn"
    ],
    entry_points={
        "console_scripts": [
            "rag_project=src.main:main",
        ],
    },
    author="Your Name",
    description="A scalable RAG pipeline using LangChain, LlamaIndex, and Melvin as a vector database.",
    license="MIT",
)
