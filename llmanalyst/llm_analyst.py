# llmanalyst/llm_analyst.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

template = """[INST]  You are provided with small portion of a pandas DataFrames:

{dataframes}

[PYTHON]
import pandas as pd

df = pd.read_csv(uploaded_file)

def analyze_data(df):
  # 1. Do not import any dataset. Use the df above only. Don't change the path.
  # 2. Carefully Analyze the given dataframe. Do not make mistakes in the row and column names of the dataframes.
  # 3. Update the template to solve the coding problem correctly. The solution must be accurate and not cause exceptions.
  # 4. Return the solution. But if the user asks to plot a graph, return plt.gcf(). Do not use plt.show().

[/PYTHON]

Please update the given analyze_data(df) function to solve the below instruction accurately. Please wrap your code answer using ```:
{prompt}
[/INST]
"""

class HuggingfaceAnalyst:
    def __init__(self, model_name):
        self.load_llm(model_name)

    def load_llm(self, model_name):
        model_name_or_path = model_name
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=False, revision="main")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate.from_template(template=template)

        self.chain = LLMChain(llm=llm, prompt=prompt)

    def process_result(self, input):
        pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
        matches = re.findall(pattern, input, re.DOTALL)

        if matches:
            extracted_code = matches[0]
            print("Extracted Python code:")
            print(extracted_code, "\n")
            return extracted_code
        else:
            return "Sorry, please ask the question again"
        

    def conversational_chat(self, query, df):
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)
        return result

    def query(self, query, df):
        result = self.conversational_chat(self, query, df)
        try:
            exec(result)
            print(analyze_data(df))
        except:
            print("Code not found")


class OpenAIGPTAnalyst:
    def __init__(self, api_key):
        self.api_key = api_key
        self.max_tokens = 512
        self.temperature = 0.7
        self.chain = self.create_chain()

    def create_chain(self):
        llm = OpenAI(openai_api_key=self.api_key)

        chain = LLMChain(
            model=llm, 
            prompt=PromptTemplate.from_template(template=self.prompt)
        )

        return chain

    def process_result(self, input):
        code_pattern = r'\[CODE\](.*?)\[/CODE\]'
        matches = re.findall(code_pattern, input, re.DOTALL)

        if matches:
            extracted_code = matches[0]
            print("Extracted code:")
            print(extracted_code, "\n")
            return extracted_code
        else:
            return "Sorry, please ask the question again"

    def conversational_chat(self, query, df):
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)
        return result
