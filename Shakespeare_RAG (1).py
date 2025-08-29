#!/usr/bin/env python
# coding: utf-8

# In[14]:


from pathlib import Path
from itertools import chain


# In[ ]:





# In[15]:


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama.llms import OllamaLLM


# ## Creating the database
# 

# In[16]:


# Step 1: Using the TextLoader and REcursiveCharacterTextSplitter, load and chunk all of the documents in data/shakespeare
# NOTE: This same method can be used with the PyPDFLoader to load a set of journal articles

chunks = None


# In[17]:


# Load source documents
source_docs = Path("../data/shakespeare").glob("*.txt")
docs = [*chain.from_iterable(TextLoader(d).load() for d in source_docs)]

# Split text to chunks.  The chunks overlap to preserve context with each other.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print(chunks[10])

print(len(chunks))


# In[18]:


# Step 2: instantiate the HuggingFaceEmbeddings model 'sentence-transformers/all-MiniLM-L6-v2' 
embedding_function = None


# In[19]:


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


# In[7]:


# Step 3: Instantiate a Chroma persistant document store 
# NOTE: Loading all of the documents can take some time, testing with a subset of the chunks can help with testing
#Only run 1 time, because otherwise yo'll get everything in double, which can fuck up with the LLM

vectorstore = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db_nccn2")
print(f"Number of chunks loaded: {vectorstore._collection.count()}")

# The vector store is "persistant" in that you can create it once and use it in multiple pipelines, even in many different notebooks/applications.


# ## Using the database

# In[20]:


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# In[21]:


# Step 1: Instantiate an OLLAMA llama3-chatqa:8b or mistral model
llm = None


# In[22]:


## Define which LLM to use
llm = OllamaLLM(model="mistral:latest", temperature=0)


# In[23]:


# Step 2: Create a prompt template.
# Remove "<<<SYSTEM INSTRUCTIONS HERE>>>" and replace it with your own system instructions

# It is a *template* so the parts in curly braces will be filled in later.
# The langchain pipeline will actually do the filling, the named parts of the prompt
# are used as keyword arguments in other parts

base_prompt = None
prompt_template  = None


# In[24]:


## Setup the prompt for querying the LLM
# It is a *template* so the parts in curly braces will be filled in later.
# The langchain pipeline will actually do the filling, the named parts of the prompt
# are used as keyword arguments in other parts

base_prompt = """
1. You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful.
2. If you don't know the answer, just say "I don't know" and stop trying to answer.
3. Use the information provided in the context to create the answer at the end.
4. The answer should be brief, limit to about 30 words.

context : {context}

question: {question}

answer:
"""

prompt_template  = PromptTemplate.from_template(base_prompt)


# In[25]:


## Connect to the vector database

# This *technically* optional since we are in the same notebook that *made* the vector DB.
# But its good practice.  What matters is that you use the same ebmedding model 
# and instantiate the same vector store pointing at the same cache.
# If it was not the same notebook that made the vector store, these lines
# would be required.

embedding_function= None
vector_db = None

# This wraps the vector database for use in a langchain pipeline
retriever = vector_db.as_retriever()


# In[26]:


embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db_nccn2", embedding_function=embedding_function)

# This wraps the vector database for use in a langchain pipeline
retriever = vector_db.as_retriever()


# In[27]:


## Invoke with the RAG pipeline via RetrievaQA
# # the chain_type_kwargs passes the prompt template we passed in earlier

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt_template})

query = None
output = qa_chain.invoke(query)

print(output['result'])
print()
print(output['source_documents'])


# In[28]:


## Invoke with the RAG pipeline via RetrievaQA
# # the chain_type_kwargs passes the prompt template we passed in earlier

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt_template})

query = {"query": "What do the collected works of Shakespeare tell us about how to live a good life?"}

output = qa_chain.invoke(query)

print(output['result'])
print()
print(output['source_documents'])


# In[29]:


# For contrast...here is the answer without the RAG context
llm_chain = prompt_template | llm 
query2 = {"question": query["query"], "context":""}
#llm_chain.invoke(query2)

for chunk in llm_chain.stream(query2):
  print(chunk, end="", flush=True)


# ## Next Steps:
# 1. Download your own collection of documents and work them through the pipeline.
# 2. Apply the principles discussed around prompt engineering.
# 3. Go back to the DataHandling example.  Find a relevant paper and combine the tabular and the paper to investigate relationships between them.

# In[ ]:




