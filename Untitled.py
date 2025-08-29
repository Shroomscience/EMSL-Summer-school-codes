#!/usr/bin/env python
# coding: utf-8

# In[11]:


from pathlib import Path
from itertools import chain


# In[12]:


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama.llms import OllamaLLM


# In[13]:


source_docs = Path("data/pdfs").glob("*.pdf")
docs = [*chain.from_iterable(PyPDFLoader(str(d)).load() for d in source_docs)]

# Split text into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)


# In[14]:


print(len(chunks))



# In[15]:


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


# In[16]:


vectorstore = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db_nccn3")
print(f"Number of chunks loaded: {vectorstore._collection.count()}")


# In[17]:


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# In[18]:


llm = OllamaLLM(model="mistral:latest", temperature=0)


# In[19]:


base_prompt = None
prompt_template  = None


# In[20]:


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


# In[21]:


embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db_nccn2", embedding_function=embedding_function)

# This wraps the vector database for use in a langchain pipeline
retriever = vector_db.as_retriever()


# In[22]:


qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt_template})

query = None
output = qa_chain.invoke(query)

print(output['result'])
print()
print(output['source_documents'])


# In[23]:


qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt_template})

query = {"query": "What is the most interesting data f?"}

output = qa_chain.invoke(query)

print(output['result'])
print()
print(output['source_documents'])


# In[ ]:




