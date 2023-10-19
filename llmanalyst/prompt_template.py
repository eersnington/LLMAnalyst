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
