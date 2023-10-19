
import re
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from streamlit_chat import message
from llmanalyst.llm_analyst import HuggingfaceAnalyst


# Load the llama model
@st.cache_resource
def get_llm(name):
    llm = HuggingfaceAnalyst(name)
    return llm


def process_result(input):
    pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'

    matches = re.findall(pattern, input, re.DOTALL)

    if matches:
        extracted_code = matches[0]
        print("Extracted Python code:")
        print(extracted_code, "\n")
        return (extracted_code)
    else:
        return "Sorry, please ask the question again"


st.title("Llama2 Chat CSV - 🦜🦙")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")
llm = get_llm("TheBloke/CodeLlama-13B-Instruct-GPTQ")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [
            "Hello ! Ask me(LLAMA2) about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input(
                "Query:", placeholder="Talk to csv data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = llm.conversational_chat(user_input, df)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                if i == 0:
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(
                        i), avatar_style="thumbs")
                else:
                    try:
                        result = exec(st.session_state["generated"][i])
                        message(st.session_state["past"][i], is_user=True, key=str(
                            i) + '_user', avatar_style="big-smile")
                        message("Here is the output:", key=str(
                            i), avatar_style="thumbs")
                        st.write(analyze_data(df))
                    except Exception as e:
                        print(e)
                        message(st.session_state["past"][i], is_user=True, key=str(
                            i) + 'e_user', avatar_style="big-smile")
                        message("Sorry, I couldn't process the output. Please query again.", key=str(
                            i)+'e', avatar_style="thumbs")
