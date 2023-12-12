# llmanalyst/main.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .prompt_template import template1, template2
import re
import matplotlib.pyplot as plt


class HuggingfaceAnalyst:
    def __init__(self, model_name, max_new_tokens=512, temperature=0.1):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_llm(model_name)

    def load_llm(self, model_name):
        model_name_or_path = model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", trust_remote_code=False, revision="main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate.from_template(template=template1)

        self.chain = LLMChain(llm=llm, prompt=prompt)

    def process_result(self, input):
        pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
        matches = re.findall(pattern, input, re.DOTALL)

        if matches:
            extracted_code = matches[0]
            return extracted_code
        else:
            return "Sorry, please ask the question again"

    def query_to_code(self, query, df):
        """
        Returns the code for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)
        return result

    def query_to_chat(self, query, df):
        """
        Prints the processed result for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)

        try:
            exec(result)
            print(analyze_data(df))
        except:
            print("Error in executing Code, trying again...")


class OpenAIGPTAnalyst:
    def __init__(self, api_key, max_tokens=512, temperature=0.1):
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chain = self.create_chain()

    def create_chain(self):
        llm = OpenAI(openai_api_key=self.api_key,
                     max_tokens=self.max_tokens, temperature=self.temperature)

        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(template=template2)
        )

        return chain

    def process_result(self, input):
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, input, re.DOTALL)

        if matches:
            extracted_code = matches[0]
            return extracted_code
        else:
            return "Sorry, couldn't process your question."

    def query_to_code(self, query, df):
        """
        Returns the code for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)
        return result

    def query_to_chat(self, query, df):
        """
        Prints the processed result for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)

        try:
            exec(result)
            print(analyze_data(df))
        except:
            print("Error in executing Code, trying again...")


class GooglePalmAnalyst:
    def __init__(self, api_key, max_tokens=512, temperature=0.7):
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chain = self.create_chain()

    def create_chain(self):
        llm = GooglePalm(google_api_key=self.api_key,
                         max_tokens=self.max_tokens, temperature=self.temperature)

        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(template=template2)
        )

        return chain

    def process_result(self, input):
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, input, re.DOTALL)

        if matches:
            extracted_code = matches[0]
            return extracted_code
        else:
            return "Sorry, couldn't process your question."

    def query_to_code(self, query, df):
        """
        Returns the code for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)
        return result

    def query_to_chat(self, query, df):
        """
        Prints the processed result for the given query
        """
        if self.chain is None:
            raise Exception("Chain is not created.")

        result = self.chain.run({'dataframes': df.head(), 'prompt': query})
        result = self.process_result(result)

        try:
            print(result)
            exec(result)

            # Get locals after executing the code
            local_vars = locals()

            # Access the function from the local namespace
            analyze_data = local_vars.get('analyze_data')

            if analyze_data:
                result = analyze_data(df)
            else:
                print("Function not found")
        except Exception as e:
            print("An error occurred during code execution:\n", e)
