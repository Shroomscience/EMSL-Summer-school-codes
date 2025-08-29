#!/usr/bin/env python
# coding: utf-8

# In[20]:


from pathlib import Path


# In[21]:


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama.llms import OllamaLLM

from langchain.prompts import PromptTemplate


# In[22]:


## Step 1: Load the document
# Hint: Use th ePyPDFLoader
doc = PyPDFLoader("../data/Advances in fungal sugar transport.pdf").load()


# In[23]:


# Step 2: Create a prompt template.
# Remove "<<<SYSTEM INSTRUCTIONS HERE>>>" and replace it with your own system instructions

# It is a *template* so the parts in curly braces will be filled in later.
# The langchain pipeline will actually do the filling, the named parts of the prompt
# are used as keyword arguments in other parts

base_prompt = """
1. You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful.
2. If you don't know the answer you just answer "I don't know" and stop trying to answer. 
3. Use information provided in the context to generate the answer.
4. The answer should be brief.

document : {document}

question: {question}

answer:
"""

prompt_template  = PromptTemplate.from_template(base_prompt)


# In[28]:


# Step 3: Create an OllamaLLM 'gemma3' model with low temperature.
# Note: If desired, for efficiency in multiple queries, include the argument "keep_alive=30".
llm = OllamaLLM(model="gemma3:latest", temperature=0.3, keep_alive=30)


# In[29]:


# step 4: Join the template and model into a chain
llm_chain = None


# In[30]:


llm_chain = prompt_template | llm 


# In[31]:


# Step 5: Invoke the chain with a question and the document
# Ask a variety of questions. Try starting off with a generic summarization question and get more specific.

question= "What's the most interesting information on the data?"
#non-streaming version
print(llm_chain.invoke({"question": question, "document": doc}))

#streaming version
for chunk in llm_chain.stream({"question": question, "document": doc}):
  print(chunk, end="", flush=True)


# ## Next steps:
# 
# After getting the full pipeline working, here are some other things to try
# 1. Try a variety of queries in the same document.  How does modifying the system prompt and varying the details of the question impact the output?
# 2. Change to using a different document
# 3. Change to using a document in a different format (for example, process "feynman problems to solve.txt" with the TextLoader class)

# In[ ]:




