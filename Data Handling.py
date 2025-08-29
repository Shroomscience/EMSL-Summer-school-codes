#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path 
from langchain import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd


# In[2]:


# Step 1: Read in the data with pandas and create a smaller subset for demonstration purposes

# Read in the raw source data from the ../data/merged_soil_data_expanded.xlsx file
df = None

#select just the columns ['sample_name','density','water_content','soil_type','geo_loc_name','cur_land_use']
df = None

#sample a random set of 20 rows from the dataframe
df = None


# In[3]:


# Read in the raw source data from the ../data/merged_soil_data_expanded.xlsx file
df = pd.read_excel(Path("../data/merged_soil_data_expanded.xlsx"), sheet_name="in")

#select just the columns ['sample_name','density','water_content','soil_type','geo_loc_name','cur_land_use']
df = df[['sample_name','density','water_content','soil_type','geo_loc_name','cur_land_use']]

#sample a random set of 10 rows from the dataframe
df = df.sample(10)

df.head()


# In[4]:


#convert the data frame into a string so that it can be parsed by the LLM
context_table = df.to_string()

print(context_table)


# In[5]:


# For reference: If loading directly from an excel file, that is also supported
# this is will load the entire dataset

#from langchain_community.document_loaders import  UnstructuredExcelLoader

#loader = UnstructuredExcelLoader(Path("../data/merged_soil_data_expanded.xlsx"))
#context_table = loader.load()

#context_table


# In[6]:


# Step 2: Instantiate the gemma3 ollama model with temperature 0.3.  
# If desired, for efficiency in multiple queries, include the argument "keep_alive=30".
# You can also include the argument num_predict=250. This will limit the number of tokens the LLM can output to prevent long running responses.
# By default ollama unloads models quickly to save system resources.  This will keep the model resident in memory longer.

llm = None


# In[7]:


# Load Ollama model (you could use models like 'gemma3' or a domain-specific model)
llm = OllamaLLM(model="gemma3:latest", temperature=0.3, keep_alive=30,num_predict=250)


# In[8]:


# Step 3 define a prompt template
# Define a prompt template.  The template defines the inputs expected and makes it easier to re-use the pipeline with different questions or data.
# The prompt below is very sparse and can be expanded on.  The important parts are the variables defined in curly braces "table" and "question"
# These are filled in when a lanchain pipeline is invoked

base_prompt = """
You are a helper.

Here is a data table: {table}

Answer the following question: {question}"
"""

prompt = PromptTemplate(template= base_prompt)


# In[9]:


# Step 4: Create a langchain pipeline
chain = prompt | llm 


# In[10]:


# Step 5: Ask questions using the table
question = None

#to print full response at once after the LLM generates it
response = chain.invoke({"table": context_table, "question": question})
print(response)

#to print each each invidual token as it is generated
for chunk in chain.stream({"table": context_table, "question": question}):
  print(chunk, end="", flush=True)


# # Next Steps
# 
# Try a variety of questions.  Ask things about the columns of the table or the table as a whole.
# 
# Some example topics:
# - What do each of the columns contain
# - What are the unique values of the variables
# - Define elements found in the table
# - Find specific rows of the table for a given criteria (e.g. most water)

# In[11]:


question = "Which are the variables in the table? Please describe them"
for chunk in chain.stream({"table": context_table, "question": question}):
  print(chunk, end="", flush=True)


# In[12]:


question = "Which states were the samples collected from?"
for chunk in chain.stream({"table": context_table, "question": question}):
  print(chunk, end="", flush=True)


# In[13]:


question = "Which sample has the most water content?"
for chunk in chain.stream({"table": context_table, "question": question}):
  print(chunk, end="", flush=True)


# In[ ]:




