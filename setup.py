from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
     name="llmanalyst",
    version="0.0.1",
    description="Talk to your CSV data with your huggingface llm models",
    packages=find_packages(where="llmanalyst"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eersnington/LLMAnalyst",
    author="Sree Narayanan",
    author_email="sreeaadhi07@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["transformers>=4.32.0","optimum>=1.12.0", "langchain",  "streamlit", "streamlit-chat", "pandas", "matplotlib"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)