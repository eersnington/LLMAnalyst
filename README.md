# Language Model Analyst

The Language Model Analyst is a Python package and Streamlit app that enables natural language generation and analysis using HuggingFace-based language models (LLMs) and OpenAI GPT-3. This package and app are designed to simplify and streamline interactions with these powerful language models.

## Usage

### Package: LLMAnalyst

The `LLMAnalyst` class allows you to interact with HuggingFace-based language models. Follow these steps to use the package:

1. Install the package:

    ```bash
    pip install git+https://github.com/eersnington/LLMAnalyst.git

    # For using GPTQ models
    pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  
    ```

2. Import and create an instance of LLMAnalyst:

    ```python
    from LLMAnalyst import LLMAnalyst
    llm_analyst = LLMAnalyst("TheBloke/CodeLlama-13B-Instruct-GPTQ")
    ```

3. 
    ```python
    query = "How many number of rows are there?"
    df = pd.read_csv("data.csv")
    result = llm_analyst.conversational_chat(query, df)
    ```


The OpenAIGPTAnalyst class enables interaction with the OpenAI GPT-3 model. Follow these steps to use the package:


1. Import and create an instance of OpenAIGPTAnalyst:

    ```python
    from LLMAnalyst import OpenAIGPTAnalyst

    openai_analyst = OpenAIGPTAnalyst(
        api_key='YOUR_OPENAI_API_KEY'
    )
    ```

2. Communicate with GPT-3:

    ```python
    query = "Calculate the mean of monthly sold data."
    df = pd.read_csv("data.csv")
    result = openai_analyst.conversational_chat(query, your_dataframe)
    ```
