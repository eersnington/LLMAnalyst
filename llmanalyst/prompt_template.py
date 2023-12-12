# FOR LLAMA MODELS
template1 = """[INST] You are provided with small portion of a pandas DataFrames:

{dataframes}

Write a function that will execute the below question / instruction from a user:
User: {prompt}

The function should return the solution to the question / instruction. If the user asks to plot a graph, return plt.gcf(). Do not use plt.show().

[PYTHON]
def analyze_data(df):
  # 1. Do not import any dataset. Use 'df' variable from the function parameter only.
  # 2. Carefully Analyze the given dataframe. Do not make mistakes in the row and column names of the dataframes.
  # 3. Update the template to solve the coding problem correctly. The solution must be accurate and not cause exceptions.
  # 4. Return the solution. But if the user asks to plot a graph, return plt.gcf(). Do not use plt.show().

[/PYTHON]

Please update the given analyze_data(df) function to solve the given question accurately. Please wrap your code answer using ```
[/INST]
"""

# FOR OPENAI GPT / GOOGLE PALM
template2 = """[INST]  You are provided with small portion of a pandas DataFrames:

{dataframes}

Write a function that will execute the below question / instruction from a user:
User: {prompt}

The function should return the solution to the question / instruction. If the user asks to plot a graph, return plt.gcf(). Do not use plt.show().

```python
def analyze_data(df):
  # 1. Do not import any dataset. Use 'df' variable from the function parameter only.
  # 2. Carefully Analyze the given dataframe. Do not make mistakes in the row and column names of the dataframes.
  # 3. Update the template to solve the coding problem correctly. The solution must be accurate and not cause exceptions.
  # 4. Return the solution. But if the user asks to plot a graph, return plt.gcf(). Do not use plt.show().

```

Please update the given analyze_data(df) function to solve the given question accurately. Please wrap your code answer using ```
[/INST]
"""
